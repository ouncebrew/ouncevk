/* SPDX-License-Identifier: MIT */
#ifndef WSI_SWITCH_NATIVE_H
#define WSI_SWITCH_NATIVE_H
#include <stdbool.h>
#include <stdint.h>

/* OunceVK's state for the window of a VI surface. Applications pass the
 * default NWindow (nwindowGetDefault) to vkCreateViSurfaceNN, as on Switch 1,
 * or a window of libounce-nx's nwindowCreateForDisplay. The libounce-nx
 * NWindow owns its VI2 layer (nwindowGetVi2Layer). The WSI
 * (wsi_switch_native.c) keeps this state for each such window. */
#define WSI_NATIVE_VI_WINDOW_MAGIC UINT64_C(0x4e58564b56493231)
/* At most this many retired swapchains wait for their images to leave the
 * layer. With three images each and three for the current swapchain, the
 * layer holds 15 images. */
#define WSI_NATIVE_HANDOVER_MAX 4

struct wsi_native_vi_window {
   uint64_t magic;
   /* The NWindow (NWindow *) whose layer this window presents to. */
   void *nwindow;
   uint32_t width, height;
   bool claimed, retired;
   /* Set only after the swapchain retires its saved releases and every
    * imported image. Layer teardown can use this as one prerequisite. It
    * does not show that the layer or the GPU owner is complete. */
   bool image_retirement_complete;
   uint64_t layer_token;
   /* Swapchains that an oldSwapchain handover retired while their images
    * were still registered on the layer. The images stay registered until
    * a later present supersedes them and their saved releases arrive. No
    * whole-layer disable is sent, so the layer does not go blank.
    */
   void *handover[WSI_NATIVE_HANDOVER_MAX];
   uint32_t handover_count;
   void (*observe)(void *context, uint32_t stage, uint32_t value);
   void *observe_context;
};
static inline void
wsi_native_vi_window_init(struct wsi_native_vi_window *window,
                          uint32_t width, uint32_t height)
{
   *window = (struct wsi_native_vi_window) {
      .magic = WSI_NATIVE_VI_WINDOW_MAGIC, .width = width, .height = height,
   };
}
struct wsi_native_vi_import {
   uint32_t memory_id, image_id;
   bool remote_use;
};
#endif
