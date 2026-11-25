import torch
import torchvision.models.segmentation as seg
from torch.utils.data import DataLoader

from loopxdino.data import VOCSegmentationDataset
from loopxdino.metrics import SegmentationEvaluator

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = seg.deeplabv3_resnet50(weights='DEFAULT').to(device).eval()
    ds = VOCSegmentationDataset(root='data/pascal_voc', image_set='val', img_size=512, download=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    ev = SegmentationEvaluator(num_classes=21, ignore_index=255)

    count = 0
    with torch.no_grad():
        for imgs, _, targets in dl:
            imgs = imgs.to(device)
            targets = targets.to(device)
            out = m(imgs)['out']
            ev.update(out, targets)
            count += imgs.shape[0]
            if count >= 400:
                break

    summary = ev.summary()
    print(f"DeepLabV3-ResNet50 (42.00M params): mIoU = {summary['mIoU']:.2f}%, bIoU = {summary['Boundary_IoU']:.2f}%", flush=True)
