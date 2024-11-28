import Augmentor

p = Augmentor.Pipeline("jpg/temp")

# p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)
# p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

p.rotate90(probability=0.5)
p.rotate270(probability=0.5)
p.flip_left_right(probability=0.8)
p.flip_top_bottom(probability=0.3)
p.crop_random(probability=1, percentage_area=0.5)
p.resize(probability=1.0, width=120, height=120)
p.ti
p.sample(10)

# p.process()











