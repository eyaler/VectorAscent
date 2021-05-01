"""
Author: Ajay Jain
Generate SVG vector images from a caption
Based on https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py
"""
import argparse
import math
import random
import numpy as np
import torch
import os
import skimage
import skimage.io
import warnings
import pydiffvg
import clip_utils

gamma = 1.0
radius = 0.05

def main(args):
    if args.seed:
      np.random.seed(args.seed)
      random.seed(args.seed)
      torch.manual_seed(args.seed)

    pydiffvg.set_print_timing(False)

    outdir = os.path.join(args.results_dir, args.prompt, args.subdir)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    canvas_width, canvas_height = 224, 224
    margin = args.initial_margin
    total_paths = args.open_paths+args.closed_paths
    step = min(args.step, total_paths)
    if step==0:
      step = args.total_paths
    
    fill_color = None
    stroke_color = None
    shapes = []
    shape_groups = []
    losses = []
    tt=0
    for num_paths in range(step,total_paths+1, step):
      for i in range(num_paths-step, num_paths):
        num_segments = random.randint(1,args.extra_segments+1)
        p0 = (margin+random.random()*(1-2*margin), margin+random.random()*(1-2*margin))
        points = [p0]
        is_closed = i>=args.open_paths
        if is_closed: 
            num_segments += 2
        for j in range(num_segments):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            if is_closed and j < num_segments - 1 or not is_closed:
                points.append(p3)
                p0 = p3        
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        stroke_width = torch.tensor(1.0)
        color = torch.tensor([random.random(),
                              random.random(),
                              random.random(),
                              random.random()])
        num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
        path = pydiffvg.Path(num_control_points = num_control_points,
                            points = points,
                            stroke_width = stroke_width,
                            is_closed = is_closed)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = color if is_closed else None,
                                            stroke_color = None if is_closed else color)
        shape_groups.append(path_group)
      
      scene_args = pydiffvg.RenderFunction.serialize_scene(\
          canvas_width, canvas_height, shapes, shape_groups)
      
      render = pydiffvg.RenderFunction.apply
      img = render(canvas_width, # width
                  canvas_height, # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'init.png'), gamma=gamma)

      points_vars = []
      stroke_width_vars = []
      color_vars = []
      for path in shapes:
          path.points.requires_grad = True
          points_vars.append(path.points)
          if not path.is_closed:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
      for group in shape_groups:
          if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
          else:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

      # Embed prompt
      text_features = clip_utils.embed_text(args.prompt)

      # Optimize
      points_optim = torch.optim.Adam(points_vars, lr=args.points_lr)
      if len(stroke_width_vars) > 0:
          width_optim = torch.optim.Adam(stroke_width_vars, lr=args.width_lr)
      color_optim = torch.optim.Adam(color_vars, lr=args.color_lr)
      # Adam iterations.
      final=False
      this_step_iters = max(1,round(args.num_iter*step/total_paths))
      if num_paths+step>total_paths:
          final=True
          this_step_iters += args.extra_iter
      for t in range(this_step_iters):
          points_optim.zero_grad()
          if len(stroke_width_vars) > 0:
              width_optim.zero_grad()
          color_optim.zero_grad()
          # Forward pass: render the image.
          scene_args = pydiffvg.RenderFunction.serialize_scene(\
              canvas_width, canvas_height, shapes, shape_groups)
          img = render(canvas_width, # width
                      canvas_height, # height
                      2,   # num_samples_x
                      2,   # num_samples_y
                      tt,   # seed
                      None,
                      *scene_args)
          # Save the intermediate render.
          with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'iter_{}.png'.format(tt)), gamma=gamma)
          image_features = clip_utils.embed_image(img)
          loss = -torch.cosine_similarity(text_features, image_features, dim=-1).mean()
          
          # Backpropagate the gradients.
          loss.backward()
          losses.append(loss.item())

          # Take a gradient descent step.
          points_optim.step()
          if len(stroke_width_vars) > 0:
              width_optim.step()
          color_optim.step()

          for path in shapes:
              path.points.data[:,0].clamp_(0.0, canvas_width)
              path.points.data[:,1].clamp_(0.0, canvas_height)
              if not path.is_closed:
                  path.stroke_width.data.clamp_(1.0, args.max_width)
          for group in shape_groups:
              if group.fill_color is not None:
                  group.fill_color.data[:3].clamp_(0.0, 1.0)
                  group.fill_color.data[3].clamp_(args.min_alpha, 1.0)
              else:
                  group.stroke_color.data[:3].clamp_(0.0, 1.0)
                  group.stroke_color.data[3].clamp_(args.min_alpha, 1.0)

          if tt % 10 == 0 or final and t == this_step_iters - 1:
              print('%d loss=%.3f'%(tt, 1+losses[-1]))
              pydiffvg.save_svg(os.path.join(outdir, 'iter_{}.svg'.format(tt)),
                                canvas_width, canvas_height, shapes, shape_groups)
              clip_utils.plot_losses(losses, outdir)
          tt += 1
    
    # Render the final result.
    img = render(args.final_px, # width
                 args.final_px, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'final.png'), gamma=gamma)
    # Convert the intermediate renderings to a video with a white background.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        os.path.join(outdir, "iter_%d.png"), "-vb", "20M", "-filter_complex",
        "color=white,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-movflags", "+faststart",
        os.path.join(outdir, "out.mp4")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="text to use for image generation loss")
    parser.add_argument('--results_dir', default='results/text_to_painting')
    parser.add_argument('--subdir', default='default')
    parser.add_argument("--open_paths", type=int, default=512)
    parser.add_argument("--closed_paths", type=int, default=512)
    parser.add_argument("--extra_segments", type=int, default=2)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--initial_margin", type=float, default=0)
    parser.add_argument("--min_alpha", type=float, default=0)
    parser.add_argument("--final_px", type=int, default=512)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--extra_iter", type=int, default=0)
    parser.add_argument("--points_lr", type=float, default=1.0)
    parser.add_argument("--width_lr", type=float, default=0.1)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)
