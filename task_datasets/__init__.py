DATA_DIR = '/fs/nexus-scratch/kwyang3/data/'

from .ocr_datasets import ocrDataset
from .caption_datasets import NoCapsDataset, FlickrDataset, COCOCaptionDataset, COCOCaptionDatasetTest
from .kie_datasets import SROIEDataset, FUNSDDataset, POIEDataset
from .embod_datasets import EmbodiedDataset
from .cls_datasets import (ImageNetDataset, CIFAR10Dataset, CIFAR100Dataset, OxfordIIITPet,
                           Flowers102, ImageNetOption, IwildIDOption, IwildOODOption, Flowers102Option)
from .whoops import WHOOPSCaptionDataset, WHOOPSVQADataset, WHOOPSWeirdDataset
from .iwildcam_dataset import IWildCamDataset
from .vqa_datasets import (
    TextVQADataset, DocVQADataset, OCRVQADataset, STVQADataset,
    ScienceQADataset, OKVQADataset, GQADataset, VizWizDataset,
    VQAv2Dataset, VQAv1Dataset, VisdialDataset, IconQADataset,
    VSRDataset, VCR1_MCIDataset, VCR1_OCDataset, MSCOCO_MCIDataset,
    MSCOCO_OCDataset, MSCOCO_POPEDataset,MSCOCO_POPEDataset_adversarial,
    MSCOCO_POPEDataset_popular
)

from functools import partial


dataset_class_dict = {
    # Caption Datasets
    'NoCaps': NoCapsDataset,
    'Flickr': FlickrDataset,
    'MSCOCO_caption': COCOCaptionDataset,
    'MSCOCO_caption_test': COCOCaptionDatasetTest,
    # KIE Datasets
    'SROIE': SROIEDataset,
    'FUNSD': FUNSDDataset,
    'POIE': POIEDataset,
    # VQA Datasets
    'TextVQA': TextVQADataset,
    'DocVQA': DocVQADataset,
    'OCRVQA': OCRVQADataset,
    'STVQA': STVQADataset,
    'ScienceQA': ScienceQADataset,
    'OKVQA': OKVQADataset,
    'GQA': GQADataset,
    'VizWiz': VizWizDataset,
    'VQAv2': VQAv2Dataset,
    'VQAv1': VQAv1Dataset,
    'Visdial': VisdialDataset,
    'IconQA': IconQADataset,
    'VSR': VSRDataset,
    # Embodied Datasets
    "MetaWorld": partial(EmbodiedDataset, dataset_name="MetaWorld"),
    "FrankaKitchen": partial(EmbodiedDataset, dataset_name="FrankaKitchen"),
    "Minecraft": partial(EmbodiedDataset, dataset_name="Minecraft"),
    "VirtualHome": partial(EmbodiedDataset, dataset_name="VirtualHome"),
    "MinecraftPolicy": partial(EmbodiedDataset, dataset_name="MinecraftPolicy"),
    # classification
    'ImageNet': ImageNetDataset,
    'ImageNetOption': ImageNetOption,
    'IwildIDOption': IwildIDOption,
    'IwildOODOption': IwildOODOption,
    'CIFAR10': CIFAR10Dataset,
    'CIFAR100': CIFAR100Dataset,
    'OxfordIIITPet': OxfordIIITPet,
    'Flowers102': Flowers102,
    'Flowers102Option': Flowers102Option,
    # whoops
    'WHOOPSCaption': WHOOPSCaptionDataset,
    'WHOOPSVQA': WHOOPSVQADataset,
    'WHOOPSWeird': WHOOPSWeirdDataset,
    # VCR, POPE
    'VCR1_OC': VCR1_OCDataset,
    'VCR1_MCI': VCR1_MCIDataset,
    'MSCOCO_MCI': MSCOCO_MCIDataset,
    'MSCOCO_OC': MSCOCO_OCDataset,
    'MSCOCO_pope': MSCOCO_POPEDataset,
    'MSCOCO_pope_popular': MSCOCO_POPEDataset_popular,
    'MSCOCO_pope_adversarial': MSCOCO_POPEDataset_adversarial
}
