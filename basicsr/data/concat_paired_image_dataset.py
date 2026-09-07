import copy

from torch.utils import data as data

from basicsr.data.paired_image_dataset import PairedImageDataset


class ConcatPairedImageDataset(data.ConcatDataset):
    """Concatenate several paired-image datasets from one training config."""

    def __init__(self, opt):
        datasets = []
        common_keys = [
            'phase', 'scale', 'gt_size', 'use_flip', 'use_rot', 'io_backend',
            'filename_tmpl', 'mean', 'std'
        ]

        for dataset_opt in opt['datasets']:
            child_opt = copy.deepcopy(dataset_opt)
            for key in common_keys:
                if key in opt and key not in child_opt:
                    child_opt[key] = copy.deepcopy(opt[key])
            child_opt.setdefault('name', dataset_opt.get('name', opt['name']))
            datasets.append(PairedImageDataset(child_opt))

        super().__init__(datasets)
        self.opt = opt
