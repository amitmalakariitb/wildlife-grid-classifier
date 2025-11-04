import os, random, shutil
src = "data/images"
# dst = "data/images_subset"
dst="data/test_images"
os.makedirs(dst, exist_ok=True)
all_imgs = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.jpeg','.png'))]
sampled = random.sample(all_imgs, 36)
for f in sampled:
    shutil.copy(os.path.join(src,f), os.path.join(dst,f))
print("✅ Copied", len(sampled), "images to", dst)
