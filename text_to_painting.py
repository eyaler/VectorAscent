"""
Author: Ajay Jain
Generate SVG vector images from a caption
Based on https://github.com/BachiLi/diffvg/blob/master/apps/painterly_rendering.py
"""

import argparse
import math
import os
import random

import pydiffvg
import torch
import skimage
import skimage.io

import clip_utils


pydiffvg.set_print_timing(True)

gamma = 1.0
radius = 0.05

def main(args):
    outdir = os.path.join(args.results_dir, args.prompt, args.subdir)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    canvas_width, canvas_height = 224, 224
    max_width = args.max_width
    margin = args.margin
    step = min(args.step, args.num_paths)
    if step==0:
      step = args.num_paths

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    fill_color = None
    stroke_color = None
    shapes = []
    shape_groups = []
    tt=0
    for num_paths in range(step,args.num_paths+1, step):
      for i in range(num_paths-step, num_paths):
        num_segments = random.randint(args.extra_segments)+1
        points = []
        p0 = (margin+random.random()*(1-2*margin), margin+random.random()*(1-2*margin))
        if args.use_blob: 
            num_segments += 2
        for j in range(num_segments):
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.append(p1)
            points.append(p2)
            if args.use_blob and j < num_segments - 1 or not args.use_blob:
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
                            is_closed = args.use_blob)
        shapes.append(path)
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                            fill_color = color if args.use_blob else None,
                                            stroke_color = None if args.use_blob else color)
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
      pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'init.png'), gamma=gamma)

      points_vars = []
      stroke_width_vars = []
      color_vars = []
      for path in shapes:
          path.points.requires_grad = True
          points_vars.append(path.points)
      if not args.use_blob:
          for path in shapes:
              path.stroke_width.requires_grad = True
              stroke_width_vars.append(path.stroke_width)
      if args.use_blob:
          for group in shape_groups:
              group.fill_color.requires_grad = True
              color_vars.append(group.fill_color)
      else:
          for group in shape_groups:
              group.stroke_color.requires_grad = True
              color_vars.append(group.stroke_color)

      # Embed prompt
      text_features = clip_utils.embed_text(args.prompt)

      # Optimize
      losses = []
      points_optim = torch.optim.Adam(points_vars, lr=args.points_lr)
      if len(stroke_width_vars) > 0:
          width_optim = torch.optim.Adam(stroke_width_vars, lr=args.width_lr)
      color_optim = torch.optim.Adam(color_vars, lr=args.color_lr)
      # Adam iterations.
      this_step_iters = max(1,round(args.num_iter*step/args.num_paths))
      if num_paths+step>args.num_paths:
          this_step_iters += args.iter_extra
      for t in range(this_step_iters):
          print('iteration:', tt)
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
          pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'iter_{}.png'.format(tt)), gamma=gamma)
          image_features = clip_utils.embed_image(img)
          loss = -torch.cosine_similarity(text_features, image_features, dim=-1).mean()
          print('render loss:', loss.item())
      
          # Backpropagate the gradients.
          loss.backward()
          losses.append(loss.item())

          # Take a gradient descent step.
          points_optim.step()
          for path in shapes:
              path.points.data[:,0].clamp_(0.0, canvas_width)
              path.points.data[:,1].clamp_(0.0, canvas_height)
          if len(stroke_width_vars) > 0:
              width_optim.step()
          color_optim.step()
          if len(stroke_width_vars) > 0:
              for path in shapes:
                  path.stroke_width.data.clamp_(1.0, max_width)
          if args.use_blob:
              for group in shape_groups:
                  group.fill_color.data.clamp_(0.0, 1.0)
          else:
              for group in shape_groups:
                  group.stroke_color.data[:3].clamp_(0.0, 1.0)
                  group.stroke_color.data[3].clamp_(args.min_trans, 1.0)

          if t % 10 == 0 or t == this_step_iters - 1:
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
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'final.png'), gamma=gamma)
    # Convert the intermediate renderings to a video with a white background.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        os.path.join(outdir, "iter_%d.png"), "-vb", "20M", "-filter_complex",
        "color=white,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1",
        os.path.join(outdir, "out.mp4")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="text to use for image generation loss")
    parser.add_argument('--results_dir', default='results/text_to_painting')
    parser.add_argument('--subdir', default='default')
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--extra_segments", type=int, default=2)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--margin", type=float, default=0)
    parser.add_argument("--min_trans", type=float, default=0)
    parser.add_argument("--final_px", type=int, default=512)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--iter_extra", type=int, default=0)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    parser.add_argument("--points_lr", type=float, default=1.0)
    parser.add_argument("--width_lr", type=float, default=0.1)
    parser.add_argument("--color_lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    main(args)
