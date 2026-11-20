/* SPDX-License-Identifier: MIT
 * Native VI2 WSI.
 */
#include "wsi_switch_native.h"
#include "wsi_switch_native_owner.h"
#include "wsi_common_entrypoints.h"
#include "wsi_common_private.h"
#include "vk_device.h"
#include "vk_instance.h"
#include "vk_util.h"
#include "util/os_time.h"
#include "c11/threads.h"
#include "util/simple_mtx.h"
#include "util/u_math.h"
#include <string.h>
#include <switch.h>
#include <ouncevk/profile.h>

/* Three images let a queued image stay visible while the WSI waits for the
 * layer release of a previous image. This count is a limited initial choice.
 * NATIVE_MAX_WIDTH and NATIVE_MAX_HEIGHT are 3840x2160, the docked 4K size.
 * A larger window size raises the limit. */
enum { NATIVE_IMAGE_COUNT = 3, NATIVE_SLOT_COUNT = 32,
       NATIVE_MAX_WIDTH = 3840, NATIVE_MAX_HEIGHT = 2160,
       NATIVE_POLL_NS = 1000000,
       /* The class-0x11 release counter IDs that VI2 command d9 allocates. */
       NATIVE_RELEASE_ID_MIN = 0xc0, NATIVE_RELEASE_ID_MAX = 0x11f };
static const uint64_t NATIVE_RETIRE_WAIT_NS = UINT64_C(10000000000);
/* Acquire waits for slot availability in chunks of this length. */
static const uint64_t NATIVE_NOTIFY_CHUNK_NS = UINT64_C(1000000000);
struct native_image {
   struct wsi_image base;
   struct wsi_native_vi_import import;
   struct wsn_dependency release;
   uint8_t slot;
   bool acquired;
   /* MAILBOX: presented, waiting for the present thread. */
   bool queued;
};
struct native_chain {
   struct wsi_swapchain base;
   struct wsi_native_vi_window *window;
   uint64_t available, reserved;
   uint64_t last_counter, last_threshold;
   uint32_t release_token, next_image;
   bool counter_seen, token_seen, lost;
   /* Set when a new swapchain names this one as oldSwapchain. The window
    * then passes to the new swapchain. Images that are still registered on
    * the layer go into the window's handover list. */
   bool handed_over;
   /* Set when vkDestroySwapchainKHR comes while the chain is in the
    * handover list. The chain is freed when it leaves the list. */
   bool destroy_pending;
   VkAllocationCallbacks destroy_alloc;
   /* Protects the chain state that the MAILBOX present thread also uses:
    * slot availability and reservations, release counters and image
    * states. */
   mtx_t mutex;
   bool mutex_ready;
   /* MAILBOX (software): present keeps at most one pending image. The
    * present thread sends it with f5, then waits until the frame before it
    * is released, which shows that the new frame is on the display. A
    * newer present replaces the pending image, which can then be acquired
    * again. Only an unsubmitted frame is replaced: VI2 has no command that
    * replaces a frame after f5.
    */
   bool mailbox, thread_started, thread_stop;
   thrd_t thread;
   cnd_t cond;
   int32_t pending, last_submitted;
   uint8_t pending_producer[16];
   struct native_image images[NATIVE_IMAGE_COUNT];
};

/* The state of the default NWindow's surface. */
static struct wsi_native_vi_window native_default_window;

/* The states of the surfaces of the other NWindows that have a VI2 layer:
 * the windows of libounce-nx's nwindowCreateForDisplay, each on a display
 * that the application names (a GameShare guest's display, for example). */
#define NATIVE_DISPLAY_WINDOW_COUNT 4
static struct wsi_native_vi_window native_display_windows[NATIVE_DISPLAY_WINDOW_COUNT];
static simple_mtx_t native_display_windows_mutex = SIMPLE_MTX_INITIALIZER;

/* The token of the VI2 layer of an NWindow other than the default one, or 0
 * if the window has no layer now. Such a window has its layer from its
 * creation to nwindowClose, so the call opens nothing. */
static uint64_t
native_display_window_token(NWindow *nw)
{
   Vi2Layer *layer = NULL;
   if (!nwindowIsValid(nw) || R_FAILED(nwindowGetVi2Layer(nw, &layer)))
      return 0;
   return layer->token;
}

/* Whether no swapchain uses a state. */
static bool
native_display_state_idle(const struct wsi_native_vi_window *w)
{
   return !w->claimed && !w->handover_count;
}

/* The state of an NWindow, or NULL if the window has no VI2 layer or no
 * state is free. A state serves its NWindow address until that window is
 * closed and no swapchain uses the state. */
static struct wsi_native_vi_window *
native_window_state(NWindow *nw)
{
   if (nw == nwindowGetDefault())
      return &native_default_window;
   const uint64_t token = native_display_window_token(nw);
   if (!token)
      return NULL;

   struct wsi_native_vi_window *state = NULL;
   simple_mtx_lock(&native_display_windows_mutex);
   for (unsigned i = 0; i < NATIVE_DISPLAY_WINDOW_COUNT && !state; i++) {
      struct wsi_native_vi_window *w = &native_display_windows[i];
      if (w->magic == WSI_NATIVE_VI_WINDOW_MAGIC && w->nwindow == nw)
         state = w;
   }
   /* A new window with a new layer has the address of a closed one: a
    * quarantined state of the earlier layer does not bind the new layer. */
   if (state && state->retired && native_display_state_idle(state) &&
       state->layer_token != token)
      state->magic = 0;
   for (unsigned i = 0; i < NATIVE_DISPLAY_WINDOW_COUNT && !state; i++) {
      struct wsi_native_vi_window *w = &native_display_windows[i];
      /* Free, or the state of a window that was closed. */
      if (w->magic != WSI_NATIVE_VI_WINDOW_MAGIC ||
          (native_display_state_idle(w) && !native_display_window_token(w->nwindow)))
         state = w;
   }
   if (state && (state->magic != WSI_NATIVE_VI_WINDOW_MAGIC || state->nwindow != nw)) {
      wsi_native_vi_window_init(state, 0, 0);
      state->nwindow = nw;
   }
   simple_mtx_unlock(&native_display_windows_mutex);
   return state;
}

/* The window of a VI surface on an NWindow with a VI2 layer. Its extent
 * follows nwindowGetDimensions until a swapchain claims it. */
static struct wsi_native_vi_window *
native_window(VkIcdSurfaceBase *base)
{
   NWindow *nw = ((VkIcdSurfaceVi *)base)->window;
   u32 width = 0, height = 0;
   if (!nw || R_FAILED(nwindowGetDimensions(nw, &width, &height)) || !width || !height)
      return NULL;
   struct wsi_native_vi_window *w = native_window_state(nw);
   if (!w)
      return NULL;
   if (w->magic != WSI_NATIVE_VI_WINDOW_MAGIC) {
      wsi_native_vi_window_init(w, width, height);
      w->nwindow = nw;
   } else if (!w->claimed) {
      w->width = width;
      w->height = height;
   }
   return w->retired ? NULL : w;
}

/* The window's size now, which can differ from a claiming swapchain's
 * (w->width, w->height): libounce-nx's nwindowSetModeDimensions gives the
 * window one size handheld and another docked. */
static bool
native_window_size(const struct wsi_native_vi_window *w,
                   uint32_t *width, uint32_t *height)
{
   u32 current_width = 0, current_height = 0;
   if (R_FAILED(nwindowGetDimensions(w->nwindow, &current_width, &current_height)) ||
       !current_width || !current_height)
      return false;
   *width = current_width;
   *height = current_height;
   return true;
}

/* Acquire and present return VK_SUBOPTIMAL_KHR when the window size does not
 * match the swapchain size. The application then recreates the swapchain, as
 * for a resized window. The swapchain continues to present. */
static VkResult
native_size_result(const struct wsi_native_vi_window *w)
{
   uint32_t width, height;
   if (native_window_size(w, &width, &height) &&
       (width != w->width || height != w->height))
      return VK_SUBOPTIMAL_KHR;
   return VK_SUCCESS;
}

static bool
surface_matches_provider(VkIcdSurfaceBase *surface,
                         const struct wsi_device *wsi)
{
   return native_window(surface) && wsi->vi.native_open;
}

static VkResult
native_support(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
                 uint32_t family, VkBool32 *supported)
{
   *supported = family == 0 && surface_matches_provider(surface, wsi);
   return VK_SUCCESS;
}

static VkResult
native_caps(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
              const void *next, VkSurfaceCapabilities2KHR *caps)
{
   struct wsi_native_vi_window *w = native_window(surface);
   if (!surface_matches_provider(surface, wsi))
      return VK_ERROR_SURFACE_LOST_KHR;
   /* The window size is the current extent. A swapchain of a different size
    * is also accepted, because the window can change size (docking) between
    * the application's query and its swapchain creation. */
   uint32_t width = w->width, height = w->height;
   native_window_size(w, &width, &height);
   const VkExtent2D min_extent = {1, 1};
   const VkExtent2D max_extent = {MAX2(width, NATIVE_MAX_WIDTH),
                                  MAX2(height, NATIVE_MAX_HEIGHT)};
   caps->surfaceCapabilities = (VkSurfaceCapabilitiesKHR) {
      .minImageCount = NATIVE_IMAGE_COUNT, .maxImageCount = NATIVE_IMAGE_COUNT,
      .currentExtent = {width, height},
      .minImageExtent = min_extent,
      .maxImageExtent = max_extent,
      .maxImageArrayLayers = 1,
      .supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
      .supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   vk_foreach_struct(ext, caps->pNext) {
      if (ext->sType == VK_STRUCTURE_TYPE_SURFACE_PROTECTED_CAPABILITIES_KHR)
         ((VkSurfaceProtectedCapabilitiesKHR *)ext)->supportsProtected = false;
   }
   return VK_SUCCESS;
}

static VkResult
native_formats(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
                 uint32_t *count, VkSurfaceFormatKHR *formats)
{
   if (!surface_matches_provider(surface, wsi))
      return VK_ERROR_SURFACE_LOST_KHR;
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormatKHR, out, formats, count);
   vk_outarray_append_typed(VkSurfaceFormatKHR, &out, f) {
      *f = (VkSurfaceFormatKHR){ VK_FORMAT_R8G8B8A8_UNORM,
                                 VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
   }
   return vk_outarray_status(&out);
}

static VkResult
native_formats2(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
                  const void *next, uint32_t *count, VkSurfaceFormat2KHR *formats)
{
   if (!surface_matches_provider(surface, wsi))
      return VK_ERROR_SURFACE_LOST_KHR;
   VK_OUTARRAY_MAKE_TYPED(VkSurfaceFormat2KHR, out, formats, count);
   vk_outarray_append_typed(VkSurfaceFormat2KHR, &out, f) {
      f->surfaceFormat = (VkSurfaceFormatKHR){ VK_FORMAT_R8G8B8A8_UNORM,
                                               VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
   }
   return vk_outarray_status(&out);
}

static VkResult
native_modes(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
               uint32_t *count, VkPresentModeKHR *modes)
{
   if (!surface_matches_provider(surface, wsi))
      return VK_ERROR_SURFACE_LOST_KHR;
   VK_OUTARRAY_MAKE_TYPED(VkPresentModeKHR, out, modes, count);
   vk_outarray_append_typed(VkPresentModeKHR, &out, m) { *m = VK_PRESENT_MODE_FIFO_KHR; }
   vk_outarray_append_typed(VkPresentModeKHR, &out, m) {
      *m = VK_PRESENT_MODE_MAILBOX_KHR;
   }
   return vk_outarray_status(&out);
}

static VkResult
native_rectangles(VkIcdSurfaceBase *surface, struct wsi_device *wsi,
                    uint32_t *count, VkRect2D *rectangles)
{
   struct wsi_native_vi_window *w = native_window(surface);
   if (!surface_matches_provider(surface, wsi))
      return VK_ERROR_SURFACE_LOST_KHR;
   VK_OUTARRAY_MAKE_TYPED(VkRect2D, out, rectangles, count);
   vk_outarray_append_typed(VkRect2D, &out, r) {
      *r = (VkRect2D){ {0, 0}, {w->width, w->height} };
   }
   return vk_outarray_status(&out);
}

static struct wsi_image *
native_get_image(struct wsi_swapchain *base, uint32_t index)
{
   return &((struct native_chain *)base)->images[index].base;
}

static VkResult
native_lost(struct native_chain *chain)
{
   chain->lost = true;
   return VK_ERROR_SURFACE_LOST_KHR;
}

static void
native_retire_observe(struct wsi_native_vi_window *window,
                      uint32_t stage, uint32_t value)
{
   if (window && window->observe)
      window->observe(window->observe_context, stage, value);
}

/* A counter value is a valid release only for this retained layer and
 * session, the same complemented token, full-width mode 1 and a counter that
 * does not decrease. Never use the GPU channel's producer counter, a
 * notification wake or slot availability as a release. */
static VkResult
native_release_ready(struct native_chain *chain, struct native_image *image)
{
   if (wsn_dependency_semantic_empty(&image->release)) return VK_SUCCESS;
   uint64_t counter;
   uint8_t mode;
   VkResult result = chain->base.wsi->vi.native_counter(chain->window,
      wsn_dependency_token(&image->release), &counter, &mode);
   if (result != VK_SUCCESS) return native_lost(chain);
   if (mode != 1 || counter == UINT64_MAX ||
       (chain->counter_seen && counter < chain->last_counter))
      return native_lost(chain);
   chain->counter_seen = true;
   chain->last_counter = counter;
   return counter >= wsn_dependency_threshold(&image->release) ?
      VK_SUCCESS : VK_NOT_READY;
}

static void native_handover_progress(struct wsi_native_vi_window *window);

/* Runs with the chain mutex held. It drops the mutex while it waits. */
static VkResult
native_acquire_locked(struct wsi_swapchain *base,
                      const VkAcquireNextImageInfoKHR *info, uint32_t *index)
{
   struct native_chain *chain = (struct native_chain *)base;
   if (chain->handed_over) return VK_ERROR_OUT_OF_DATE_KHR;
   if (chain->lost || chain->window->retired) return VK_ERROR_SURFACE_LOST_KHR;
   native_handover_progress(chain->window);
   const struct wsn_deadline deadline = wsn_deadline_make(os_time_get_nano(), info->timeout);
   const bool notified = base->wsi->vi.native_availability_consume &&
                         base->wsi->vi.native_availability_wait;
   const bool gpu_release = base->wsi->vi.native_release_sync != NULL;
   for (;;) {
      VkResult result;
      /* Consume this layer's availability notification before the e4 query.
       * A slot that VI2 reclaims after the query then posts a new one. */
      if (notified &&
          base->wsi->vi.native_availability_consume(chain->window) != VK_SUCCESS)
         return native_lost(chain);
      result = base->wsi->vi.native_availability(chain->window, &chain->available);
      if (result != VK_SUCCESS || chain->available >> NATIVE_SLOT_COUNT)
         return native_lost(chain);
      uint64_t available = chain->available & ~chain->reserved;
      if (available) {
         for (uint32_t n = 0; n < NATIVE_IMAGE_COUNT; n++) {
            uint32_t i = (chain->next_image + n) % NATIVE_IMAGE_COUNT;
            struct native_image *image = &chain->images[i];
            if (image->acquired || image->queued) continue;
            /* MAILBOX: the newest submitted image is released only when a
             * newer frame is on the display. A frame rendered into it would
             * wait for its own successor, so skip it until its release has
             * arrived. FIFO's order never returns it first. */
            if (chain->mailbox && (int32_t)i == chain->last_submitted) {
               result = native_release_ready(chain, image);
               if (result < 0) return result;
               if (result != VK_SUCCESS) continue;
            }
            /* With GPU release waits, the acquire semaphore or fence carries
             * the saved release (native_create_acquire_sync). Without them,
             * wait for the release here. */
            if (!gpu_release) {
               result = native_release_ready(chain, image);
               if (result < 0) return result;
               if (result != VK_SUCCESS) continue;
            }
            uint8_t slot = 0;
            while (!(available & (UINT64_C(1) << slot))) slot++;
            chain->reserved |= UINT64_C(1) << slot;
            image->slot = slot;
            image->acquired = true;
            chain->next_image = (i + 1) % NATIVE_IMAGE_COUNT;
            *index = i;
            return native_size_result(chain->window);
         }
      }
      uint64_t left = wsn_deadline_remaining(deadline, os_time_get_nano());
      if (!left) return info->timeout ? VK_TIMEOUT : VK_NOT_READY;
      if (!notified) {
         mtx_unlock(&chain->mutex);
         base->wsi->vi.native_pause(MIN2(left, NATIVE_POLL_NS));
         mtx_lock(&chain->mutex);
         continue;
      }
      /* If no VI slot is free, wait for the layer's second notification
       * index. VI2 posts it when it reclaims a slot record. Wait in
       * one-second chunks and then query again.
       * If a slot is free but the saved releases of all images are pending,
       * also wait for that notification. The release counter has no
       * notification of its own. VI2 writes the counter on the same
       * final-use path, just before the reclamation. Wake at most
       * NATIVE_POLL_NS later. The raw-77 counter check stays the release
       * test. */
      mtx_unlock(&chain->mutex);
      result = base->wsi->vi.native_availability_wait(chain->window,
         MIN2(left, available ? NATIVE_POLL_NS : NATIVE_NOTIFY_CHUNK_NS));
      mtx_lock(&chain->mutex);
      if (result != VK_SUCCESS && result != VK_TIMEOUT)
         return native_lost(chain);
      if (chain->lost) return VK_ERROR_SURFACE_LOST_KHR;
   }
}

static VkResult
native_acquire(struct wsi_swapchain *base, const VkAcquireNextImageInfoKHR *info,
               uint32_t *index)
{
   struct native_chain *chain = (struct native_chain *)base;
   const uint64_t start = armGetSystemTick();
   mtx_lock(&chain->mutex);
   const VkResult result = native_acquire_locked(base, info, index);
   mtx_unlock(&chain->mutex);
   ouncevk_profile_record(OUNCEVK_PROFILE_ACQUIRE, armGetSystemTick() - start);
   return result;
}

/* Makes the saved release of the acquired image the payload of the acquire
 * semaphore or fence. An empty release (first use) needs no payload. */
static VkResult
native_create_acquire_sync(struct wsi_swapchain *base,
                           const struct wsi_image *wsi_image,
                           enum vk_sync_features features,
                           struct vk_sync **sync_out)
{
   (void)features;
   struct native_chain *chain = (struct native_chain *)base;
   const struct native_image *image = NULL;
   for (uint32_t i = 0; i < NATIVE_IMAGE_COUNT && !image; i++)
      if (&chain->images[i].base == wsi_image)
         image = &chain->images[i];
   if (!image || !base->wsi->vi.native_release_sync ||
       wsn_dependency_semantic_empty(&image->release))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return base->wsi->vi.native_release_sync(
      vk_device_from_handle(base->device), chain->window,
      image->release.bytes, sync_out);
}

/* Sends image index with f5 and saves its release. Runs with the chain
 * mutex held. */
static VkResult
native_submit(struct native_chain *chain, uint32_t index,
              const uint8_t producer[16])
{
   struct wsi_swapchain *base = &chain->base;
   struct native_image *image = &chain->images[index];
   struct wsn_dependency release;
   VkResult result = base->wsi->vi.native_present(chain->window, image->slot,
      image->import.image_id, producer, &chain->available, release.bytes);
   if (result != VK_SUCCESS) return native_lost(chain); /* Never replay f5. */
   uint32_t token = wsn_dependency_token(&release);
   uint64_t threshold = wsn_dependency_threshold(&release);
   /* VI2 command d9 allocates counter IDs 0xc0 to 0x11f. The wire token is
    * the complement of the ID. */
   if (!wsn_dependency_canonical(&release) || wsn_dependency_mode(&release) != 1 ||
       ~token < NATIVE_RELEASE_ID_MIN || ~token > NATIVE_RELEASE_ID_MAX || threshold == UINT64_MAX ||
       (chain->token_seen && (token != chain->release_token || threshold <= chain->last_threshold)) ||
       chain->available >> NATIVE_SLOT_COUNT)
      return native_lost(chain);
   chain->token_seen = true;
   chain->release_token = token;
   chain->last_threshold = threshold;
   if (base->wsi->vi.native_startup) base->wsi->vi.native_startup(chain->window);
   image->release = release;
   chain->reserved &= ~(UINT64_C(1) << image->slot);
   return VK_SUCCESS;
}

/* The MAILBOX present thread. */
static int
native_present_thread(void *argument)
{
   struct native_chain *chain = argument;
   struct wsi_swapchain *base = &chain->base;
   mtx_lock(&chain->mutex);
   for (;;) {
      while (!chain->thread_stop && chain->pending < 0)
         cnd_wait(&chain->cond, &chain->mutex);
      if (chain->thread_stop)
         break;
      const uint32_t index = (uint32_t)chain->pending;
      const int32_t previous = chain->last_submitted;
      uint8_t producer[16];
      memcpy(producer, chain->pending_producer, sizeof(producer));
      chain->pending = -1;
      const VkResult result = native_submit(chain, index, producer);
      chain->images[index].queued = false;
      if (result != VK_SUCCESS)
         break;
      chain->last_submitted = (int32_t)index;
      /* Wait until the frame before this one is released: then this frame
       * is on the display, and the next pending frame can follow it. */
      while (previous >= 0 && (uint32_t)previous != index &&
             !chain->thread_stop) {
         const VkResult ready =
            native_release_ready(chain, &chain->images[previous]);
         if (ready != VK_NOT_READY)
            break;
         mtx_unlock(&chain->mutex);
         base->wsi->vi.native_pause(NATIVE_POLL_NS);
         mtx_lock(&chain->mutex);
      }
      if (chain->lost)
         break;
   }
   mtx_unlock(&chain->mutex);
   return 0;
}

/* Stops the MAILBOX present thread and drops an unsent pending image. */
static void
native_mailbox_stop(struct native_chain *chain)
{
   if (!chain->thread_started)
      return;
   mtx_lock(&chain->mutex);
   chain->thread_stop = true;
   cnd_signal(&chain->cond);
   mtx_unlock(&chain->mutex);
   thrd_join(chain->thread, NULL);
   chain->thread_started = false;
   if (chain->pending >= 0) {
      struct native_image *image = &chain->images[chain->pending];
      image->queued = false;
      chain->reserved &= ~(UINT64_C(1) << image->slot);
      chain->pending = -1;
   }
}

static VkResult
native_present_untimed(struct wsi_swapchain *base, uint32_t index,
                       uint64_t present_id, const VkPresentRegionKHR *damage)
{
   struct native_chain *chain = (struct native_chain *)base;
   if (chain->handed_over) return VK_ERROR_OUT_OF_DATE_KHR;
   if (chain->lost || index >= NATIVE_IMAGE_COUNT || !chain->images[index].acquired)
      return native_lost(chain);
   struct native_image *image = &chain->images[index];
   /* Common WSI submits all application wait semaphores before this fence.
    * VI2 waits for the GPU producer of the fence. The order is fence, flush,
    * then f5 with that dependency, with no CPU wait. If nothing is submitted
    * for the fence yet, the CPU waits for it. The fence keeps the resources
    * of the GPU work. Common WSI waits for the fence before it reuses the
    * fence of this image. */
   VkFence fence = base->fences[index];
   if (!fence) return native_lost(chain);
   uint8_t producer[16];
   VkResult result = base->wsi->vi.native_fence_dependency ?
      base->wsi->vi.native_fence_dependency(base->device, fence, producer) :
      VK_NOT_READY;
   if (result == VK_NOT_READY) {
      memset(producer, 0, sizeof(producer));
      result = base->wsi->WaitForFences(base->device, 1, &fence, true, UINT64_MAX);
   }
   if (result != VK_SUCCESS) { chain->lost = true; return result; }
   mtx_lock(&chain->mutex);
   if (chain->mailbox) {
      if (chain->lost) {
         mtx_unlock(&chain->mutex);
         return VK_ERROR_SURFACE_LOST_KHR;
      }
      /* Replace an unsent pending image. It can be acquired again. */
      if (chain->pending >= 0) {
         struct native_image *replaced = &chain->images[chain->pending];
         replaced->queued = false;
         chain->reserved &= ~(UINT64_C(1) << replaced->slot);
      }
      chain->pending = (int32_t)index;
      memcpy(chain->pending_producer, producer, sizeof(producer));
      image->acquired = false;
      image->queued = true;
      cnd_signal(&chain->cond);
   } else {
      result = native_submit(chain, index, producer);
      image->acquired = false;
   }
   mtx_unlock(&chain->mutex);
   if (result != VK_SUCCESS) return result;
   native_handover_progress(chain->window);
   return native_size_result(chain->window);
}

static VkResult
native_present(struct wsi_swapchain *base, uint32_t index,
               uint64_t present_id, const VkPresentRegionKHR *damage)
{
   const uint64_t start = armGetSystemTick();
   const VkResult result =
      native_present_untimed(base, index, present_id, damage);
   ouncevk_profile_record(OUNCEVK_PROFILE_PRESENT, armGetSystemTick() - start);
   return result;
}

static VkResult
native_create_memory(const struct wsi_swapchain *chain,
                       const struct wsi_image_info *info, struct wsi_image *image)
{
   VkMemoryRequirements req;
   chain->wsi->GetImageMemoryRequirements(chain->device, image->image, &req);
   /* The VI2 import maps the image on the CPU and registers it with the
    * GPU-uncached backing (native_vi_import). The display reads the image
    * directly. Use the host-coherent pool, not the GPU-cacheable pool. */
   uint32_t type = 0;
   while (type < chain->wsi->memory_props.memoryTypeCount &&
          (!(req.memoryTypeBits & (1u << type)) ||
           !(chain->wsi->memory_props.memoryTypes[type].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) type++;
   if (type == chain->wsi->memory_props.memoryTypeCount || req.size > UINT32_MAX)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   const VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = image->image,
   };
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &dedicated,
      .allocationSize = req.size, .memoryTypeIndex = type,
   };
   VkResult result = chain->wsi->AllocateMemory(chain->device, &alloc, &chain->alloc, &image->memory);
   if (result == VK_SUCCESS) {
      image->num_planes = 1;
      image->sizes[0] = req.size;
   }
   return result;
}

/* Retires every image that VI2 still uses. The order is a layer-wide
 * disabled f4, a check of each saved release, then the GSV image retirement.
 * On success, releases the window for a new swapchain. On failure,
 * quarantines the window and returns VK_ERROR_SURFACE_LOST_KHR. */
/* Protects the handover list of the window. The new swapchain's present
 * and the old swapchain's destroy can run on different threads. */
static simple_mtx_t native_handover_mutex = SIMPLE_MTX_INITIALIZER;

static void
native_chain_free(struct native_chain *chain, const VkAllocationCallbacks *alloc)
{
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++)
      if (chain->images[i].base.image)
         wsi_destroy_image(&chain->base, &chain->images[i].base);
   if (chain->mutex_ready) {
      cnd_destroy(&chain->cond);
      mtx_destroy(&chain->mutex);
   }
   wsi_swapchain_finish(&chain->base);
   vk_free(alloc, chain);
}

static bool
native_chain_registered(const struct native_chain *chain)
{
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++)
      if (chain->images[i].import.remote_use)
         return true;
   return false;
}

/* Removes the images of a retired swapchain from the layer, without a
 * whole-layer disable. Every saved release must already be qualified. */
static VkResult
native_unregister_images(struct native_chain *chain)
{
   struct wsi_swapchain *base = &chain->base;
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      struct native_image *image = &chain->images[i];
      if (image->import.remote_use &&
          base->wsi->vi.native_retire_image(base->device, image->base.memory,
                                            chain->window,
                                            &image->import) != VK_SUCCESS)
         return VK_ERROR_SURFACE_LOST_KHR;
   }
   return VK_SUCCESS;
}

/* Returns VK_SUCCESS when the saved releases of every registered image of
 * the chain are qualified, VK_NOT_READY when one is still pending. */
static VkResult
native_releases_ready(struct native_chain *chain)
{
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      struct native_image *image = &chain->images[i];
      if (!image->import.remote_use)
         continue;
      const VkResult ready = native_release_ready(chain, image);
      if (ready != VK_SUCCESS)
         return ready;
   }
   return VK_SUCCESS;
}

/* Removes entry i of the handover list. Frees the chain if its destroy
 * came already. Call with the handover mutex held. */
static void
native_handover_remove(struct wsi_native_vi_window *window, uint32_t i)
{
   struct native_chain *chain = window->handover[i];
   window->handover[i] = window->handover[--window->handover_count];
   window->handover[window->handover_count] = NULL;
   if (chain->destroy_pending)
      native_chain_free(chain, &chain->destroy_alloc);
}

/* Unregisters the images of each retired swapchain whose releases have
 * arrived. It does not wait. The current swapchain calls it at acquire and
 * after each present: a later present supersedes the old images. */
static void
native_handover_progress(struct wsi_native_vi_window *window)
{
   if (!window || !window->handover_count)
      return;
   simple_mtx_lock(&native_handover_mutex);
   for (uint32_t i = 0; i < window->handover_count;) {
      struct native_chain *old = window->handover[i];
      const VkResult ready = native_releases_ready(old);
      if (ready == VK_NOT_READY) {
         i++;
         continue;
      }
      if (ready != VK_SUCCESS || native_unregister_images(old) != VK_SUCCESS) {
         /* Keep every owner of an uncertain image. */
         window->retired = true;
         i++;
         continue;
      }
      native_retire_observe(window, 856, i);
      native_handover_remove(window, i);
   }
   simple_mtx_unlock(&native_handover_mutex);
}

static VkResult
native_retire_images(struct native_chain *chain)
{
   struct wsi_swapchain *base = &chain->base;
   struct wsi_native_vi_window *window = chain->window;
   bool retained = native_chain_registered(chain) ||
                   (window && window->handover_count);
   if (!retained)
      return VK_SUCCESS;
   native_retire_observe(window, 850, 1);
   if (!window || chain->lost || !window->layer_token ||
       !base->wsi->vi.native_disable ||
       !base->wsi->vi.native_retire_image)
      goto quarantine;
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      const struct native_image *image = &chain->images[i];
      if (image->acquired ||
          (image->import.remote_use &&
           (!image->import.memory_id || !image->import.image_id)))
         goto quarantine;
   }
   /* A disabled f4 applies to the whole layer. Submit it before the wait for
    * the saved release of the most recent image. That release can depend on
    * this update. */
   const struct wsn_deadline slot_deadline =
      wsn_deadline_make(os_time_get_nano(), NATIVE_RETIRE_WAIT_NS);
   uint64_t mask = 0;
   for (;;) {
      if (base->wsi->vi.native_availability(window, &mask) != VK_SUCCESS ||
          mask >> NATIVE_SLOT_COUNT)
         goto quarantine;
      mask &= ~chain->reserved;
      if (mask) break;
      uint64_t left = wsn_deadline_remaining(slot_deadline, os_time_get_nano());
      if (!left) goto quarantine;
      base->wsi->vi.native_pause(MIN2(left, NATIVE_POLL_NS));
   }
   const uint8_t slot = (uint8_t)__builtin_ctzll(mask);
   native_retire_observe(window, 851, slot);
   if (base->wsi->vi.native_disable(window, slot, &mask) != VK_SUCCESS ||
       mask >> NATIVE_SLOT_COUNT)
      goto quarantine; /* Never replay an uncertain f4. */
   native_retire_observe(window, 852, 1);
   const struct wsn_deadline release_deadline =
      wsn_deadline_make(os_time_get_nano(), NATIVE_RETIRE_WAIT_NS);
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      struct native_image *image = &chain->images[i];
      while (image->import.remote_use) {
         VkResult ready = native_release_ready(chain, image);
         if (ready == VK_SUCCESS) break;
         if (ready != VK_NOT_READY) goto quarantine;
         uint64_t left = wsn_deadline_remaining(release_deadline, os_time_get_nano());
         if (!left) goto quarantine;
         base->wsi->vi.native_pause(MIN2(left, NATIVE_POLL_NS));
      }
      native_retire_observe(window, 853, i);
   }
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      struct native_image *image = &chain->images[i];
      if (image->import.remote_use &&
          base->wsi->vi.native_retire_image(base->device,
             image->base.memory, window, &image->import) != VK_SUCCESS)
         goto quarantine;
      native_retire_observe(window, 854, i);
   }
   /* The disable also supersedes the images of retired swapchains that are
    * still on the layer. Wait for their releases and remove them too. */
   simple_mtx_lock(&native_handover_mutex);
   while (window->handover_count) {
      struct native_chain *old = window->handover[0];
      VkResult ready;
      while ((ready = native_releases_ready(old)) == VK_NOT_READY) {
         uint64_t left = wsn_deadline_remaining(release_deadline,
                                                os_time_get_nano());
         if (!left) break;
         base->wsi->vi.native_pause(MIN2(left, NATIVE_POLL_NS));
      }
      if (ready != VK_SUCCESS || native_unregister_images(old) != VK_SUCCESS) {
         simple_mtx_unlock(&native_handover_mutex);
         goto quarantine;
      }
      native_retire_observe(window, 857, window->handover_count);
      native_handover_remove(window, 0);
   }
   simple_mtx_unlock(&native_handover_mutex);
   window->claimed = false;
   window->image_retirement_complete = true;
   return VK_SUCCESS;
quarantine:
   if (window) window->retired = true;
   native_retire_observe(window, 859, 1);
   return VK_ERROR_SURFACE_LOST_KHR;
}

static VkResult
native_destroy(struct wsi_swapchain *base, const VkAllocationCallbacks *alloc)
{
   struct native_chain *chain = (struct native_chain *)base;
   struct wsi_native_vi_window *window = chain->window;
   native_mailbox_stop(chain);
   /* A swapchain that an oldSwapchain handover retired gave the window to
    * the new swapchain. If its images are still on the layer, it stays in
    * the handover list and is freed when it leaves it. */
   if (chain->handed_over) {
      simple_mtx_lock(&native_handover_mutex);
      bool listed = false;
      for (uint32_t i = 0; window && i < window->handover_count; i++)
         listed |= window->handover[i] == chain;
      if (listed) {
         chain->destroy_pending = true;
         chain->destroy_alloc = *alloc;
      }
      simple_mtx_unlock(&native_handover_mutex);
      if (listed)
         return VK_SUCCESS;
   } else {
      bool retained = native_chain_registered(chain) ||
                      (window && window->handover_count);
      if (retained) {
         if (native_retire_images(chain) != VK_SUCCESS)
            return VK_SUCCESS; /* quarantined: keep every owner */
      } else if (chain->window && chain->window->claimed) {
         /* A failed bootstrap can leave an unknown remote owner. */
         chain->window->retired = true;
      }
   }
   native_chain_free(chain, alloc);
   native_retire_observe(window, 855, 1);
   return VK_SUCCESS;
}

/* Handles vkCreateSwapchainKHR with oldSwapchain on the same window. The old
 * swapchain retires even if the new one fails: its acquire and present then
 * return VK_ERROR_OUT_OF_DATE_KHR, also for images that the application
 * still holds. This function waits for the present fences of the old
 * images (each covers the rendering that its present waited for). The old
 * images stay registered on the layer, so the layer keeps its last frame
 * until the new swapchain presents. The old swapchain goes into the
 * window's handover list, and native_handover_progress removes its images
 * when their releases arrive.
 */
static VkResult
native_hand_over(struct native_chain *old, struct wsi_native_vi_window *w)
{
   if (old->window != w || old->handed_over)
      return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   if (w->handover_count == WSI_NATIVE_HANDOVER_MAX)
      native_handover_progress(w);
   if (w->handover_count == WSI_NATIVE_HANDOVER_MAX)
      return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   /* An unsent MAILBOX image is dropped. */
   native_mailbox_stop(old);
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      VkFence fence = old->base.fences ? old->base.fences[i] : VK_NULL_HANDLE;
      if (fence && old->base.wsi->WaitForFences(old->base.device, 1, &fence,
                                                true, UINT64_MAX) != VK_SUCCESS)
         return VK_ERROR_DEVICE_LOST;
   }
   old->handed_over = true;
   old->lost = true;
   /* An acquired image cannot be presented any more. Its VI slot
    * reservation is local only (no f5 used it), so release it. */
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++)
      old->images[i].acquired = false;
   old->reserved = 0;
   if (native_chain_registered(old)) {
      simple_mtx_lock(&native_handover_mutex);
      w->handover[w->handover_count++] = old;
      simple_mtx_unlock(&native_handover_mutex);
   }
   w->claimed = false;
   return VK_SUCCESS;
}

static VkResult
native_create_chain(VkIcdSurfaceBase *surface, VkDevice device, struct wsi_device *wsi,
                      const VkSwapchainCreateInfoKHR *info,
                      const VkAllocationCallbacks *alloc, struct wsi_swapchain **out)
{
   struct wsi_native_vi_window *w = native_window(surface);
   if (!w) return VK_ERROR_SURFACE_LOST_KHR;
   if (info->oldSwapchain) {
      VK_FROM_HANDLE(wsi_swapchain, old_base, info->oldSwapchain);
      const VkResult handed = native_hand_over(
         (struct native_chain *)old_base, w);
      if (handed != VK_SUCCESS) return handed;
      /* The old swapchain released the window. Use its current size. */
      w = native_window(surface);
      if (!w) return VK_ERROR_SURFACE_LOST_KHR;
   }
   if (w->claimed) return VK_ERROR_NATIVE_WINDOW_IN_USE_KHR;
   uint32_t max_width = NATIVE_MAX_WIDTH, max_height = NATIVE_MAX_HEIGHT;
   native_window_size(w, &max_width, &max_height);
   max_width = MAX2(max_width, NATIVE_MAX_WIDTH);
   max_height = MAX2(max_height, NATIVE_MAX_HEIGHT);
   if (info->imageFormat != VK_FORMAT_R8G8B8A8_UNORM ||
       info->imageColorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
       (info->presentMode != VK_PRESENT_MODE_FIFO_KHR &&
        info->presentMode != VK_PRESENT_MODE_MAILBOX_KHR) ||
       info->minImageCount > NATIVE_IMAGE_COUNT || info->imageArrayLayers != 1 ||
       !info->imageExtent.width || info->imageExtent.width > max_width ||
       !info->imageExtent.height || info->imageExtent.height > max_height ||
       info->flags || !wsi->vi.native_import)
      return VK_ERROR_INITIALIZATION_FAILED;
   /* The swapchain size sets its images, crop and VI2 image layout. When it
    * differs from the window size, acquire and present return
    * VK_SUBOPTIMAL_KHR. */
   w->width = info->imageExtent.width;
   w->height = info->imageExtent.height;
   struct native_chain *chain = vk_zalloc(alloc, sizeof(*chain), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!chain) return VK_ERROR_OUT_OF_HOST_MEMORY;
   const struct wsi_cpu_image_params params = { .base.image_type = WSI_IMAGE_TYPE_CPU };
   VkResult result = wsi_swapchain_init(wsi, &chain->base, device, info, &params.base, alloc);
   if (result != VK_SUCCESS) { vk_free(alloc, chain); return result; }
   chain->window = w;
   chain->pending = -1;
   chain->last_submitted = -1;
   chain->mailbox = info->presentMode == VK_PRESENT_MODE_MAILBOX_KHR;
   if (mtx_init(&chain->mutex, mtx_plain) != thrd_success) {
      wsi_swapchain_finish(&chain->base);
      vk_free(alloc, chain);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (cnd_init(&chain->cond) != thrd_success) {
      mtx_destroy(&chain->mutex);
      wsi_swapchain_finish(&chain->base);
      vk_free(alloc, chain);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   chain->mutex_ready = true;
   chain->base.image_count = NATIVE_IMAGE_COUNT;
   chain->base.present_mode = info->presentMode;
   chain->base.destroy = native_destroy;
   chain->base.get_wsi_image = native_get_image;
   chain->base.acquire_next_image = native_acquire;
   chain->base.create_acquire_sync = native_create_acquire_sync;
   chain->base.queue_present = native_present;
   wsi_destroy_image_info(&chain->base, &chain->base.image_info);
   chain->base.blit.type = WSI_SWAPCHAIN_NO_BLIT;
   result = wsi_configure_image(&chain->base, info, 0, &chain->base.image_info);
   if (result != VK_SUCCESS) goto fail;
   chain->base.image_info.create_mem = native_create_memory;
   chain->base.image_info.finish_create = NULL;
   /* After a swapchain retires every image, nothing of this window stays on
    * its layer. Present to the current layer of the NWindow. That is a new
    * layer if the old layer was released (nwindowReleaseVi2Layer, at GPU
    * shutdown). */
   if (w->image_retirement_complete)
      w->layer_token = 0;
   w->claimed = true; /* A failed bootstrap can also create remote owners. */
   w->image_retirement_complete = false;
   result = wsi->vi.native_open(w);
   if (result != VK_SUCCESS) goto fail;
   for (unsigned i = 0; i < NATIVE_IMAGE_COUNT; i++) {
      struct native_image *image = &chain->images[i];
      result = wsi_create_image(&chain->base, &chain->base.image_info, &image->base);
      if (result != VK_SUCCESS) { memset(&image->base, 0, sizeof(image->base)); goto fail; }
      result = wsi->vi.native_import(device, image->base.image, image->base.memory, w, &image->import);
      if (result != VK_SUCCESS) goto fail;
   }
   if (chain->mailbox) {
      if (thrd_create(&chain->thread, native_present_thread, chain) !=
          thrd_success) {
         result = VK_ERROR_OUT_OF_HOST_MEMORY;
         goto fail;
      }
      chain->thread_started = true;
   }
   *out = &chain->base;
   return VK_SUCCESS;
fail:
   native_destroy(&chain->base, alloc);
   return result;
}

VkResult
wsi_switch_init_wsi(struct wsi_device *wsi, const VkAllocationCallbacks *alloc,
                     VkPhysicalDevice physical_device)
{
   struct wsi_interface *iface = vk_zalloc(alloc, sizeof(*iface), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!iface) return VK_ERROR_OUT_OF_HOST_MEMORY;
   *iface = (struct wsi_interface) {
      .get_support = native_support, .get_capabilities2 = native_caps,
      .get_formats = native_formats, .get_formats2 = native_formats2,
      .get_present_modes = native_modes, .get_present_rectangles = native_rectangles,
      .create_swapchain = native_create_chain,
   };
   wsi->wsi[VK_ICD_WSI_PLATFORM_VI] = iface;
   return VK_SUCCESS;
}

void
wsi_switch_finish_wsi(struct wsi_device *wsi, const VkAllocationCallbacks *alloc)
{
   vk_free(alloc, wsi->wsi[VK_ICD_WSI_PLATFORM_VI]);
   wsi->wsi[VK_ICD_WSI_PLATFORM_VI] = NULL;
}

VKAPI_ATTR VkResult VKAPI_CALL
wsi_CreateViSurfaceNN(VkInstance handle, const VkViSurfaceCreateInfoNN *info,
                       const VkAllocationCallbacks *alloc, VkSurfaceKHR *out)
{
   VK_FROM_HANDLE(vk_instance, instance, handle);
   VkIcdSurfaceVi candidate = { .window = info->window };
   if (info->flags || !native_window(&candidate.base))
      return VK_ERROR_SURFACE_LOST_KHR;
   VkIcdSurfaceVi *surface = vk_alloc2(&instance->alloc, alloc, sizeof(*surface), 8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!surface) return VK_ERROR_OUT_OF_HOST_MEMORY;
   surface->base.platform = VK_ICD_WSI_PLATFORM_VI;
   surface->window = info->window;
   *out = VkIcdSurfaceBase_to_handle(&surface->base);
   return VK_SUCCESS;
}
