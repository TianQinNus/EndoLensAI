# Standalone predicted polyp outlines

This retained teammate package accepts a photo and writes a native-size PNG with green predicted mask outlines, no fill or rectangular box. Width defaults2, allowed1-20; multiple regions and internal holes are retained. Empty prediction returns the original image; EXIF orientation is respected and output is RGB.

~~~bash
python3 -m pip install -r requirements.txt
python3 predict.py /absolute/path/photo.jpg -o /absolute/path/photo_outline.png
~~~

Run within this directory. Without -o, the suffix is _outline.png beside the input. Existing outputs are refused. CPU default avoids competing with training; mps/cuda may be selected only when appropriate and independently checked.

~~~python
from predict import PolypOutline
model = PolypOutline()
model.predict("photo.jpg", "photo_outline.png")
~~~

annotate(PIL_image) returns an image without saving. This tool requires its separately supplied best.pt; the B LFS export is a different architecture/format and must not be substituted.

## Original selection and protocol

The existing model_info.json records teammate run unetpp_context_20261020, epoch100, selected validation Dice0.893595; historical798 native test Dice0.718374 and IoU0.638657. Those known test entries are not an independent final holdout, and these validation settings differ from E8 B. The original selection rule is preserved rather than replacing it with another test-favorable model.

This line trained from random initialization, without an external pretrained encoder. RGB/255, bilinear320 square, native logits bilinear restore align_corners=False, sigmoid threshold0.5; inner edges are mask minus erosion. Outlines are predictions, not GT.

model.py and best.pt are checked against model_info.json SHA values. This publication changes only the CLI description into English; architecture, source/checkpoint identity and computation are preserved. Current project integration separately uses the frozen B SDK under ../endolens_b.
