from .vision import VisionDataset
from PIL import Image
import os
import os.path
from io import BytesIO


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its
            target as entry and returns a transformed version.
    """

    def __init__(self, root, ann_handler, transform=None, target_transform=None, transforms=None,
                 gs_bucket_name=None):
        super(CocoDetection, self).__init__(root, transforms, transform, target_transform)
        self.coco = ann_handler
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.process_local_initialized = False
        self.gs_bucket = None
        self.gs_bucket_name = gs_bucket_name

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        if not self.process_local_initialized and self.gs_bucket_name is not None:
            self.gs_bucket = self._get_gc_bucket(self.gs_bucket_name)
            self.process_local_initialized = True

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        image_name = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self.root, image_name)

        if self.gs_bucket is not None:
            image_blob = self.gs_bucket.get_blob(image_path)
            image_bytes = image_blob.download_as_string()
            image_file = BytesIO(image_bytes)
            img = Image.open(image_file).convert('RGB')
        else:
            img = Image.open(image_path).convert('RGB')

        target = {'image_id': img_id, 'annotations': target}
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    def _get_gc_bucket(self, bucket_name):
        """Authenticates to gcloud and returns gcloud storage bucket.

        This function needs to run gc authentication, which needs to run
        in the process that is going to use the bucket according to:

        `https://googleapis.github.io/google-cloud-python/latest/storage/index.html?highlight=thread%20safe#example-usage`

        which reads:

        'Because the storage client uses the third-party requests library
         by default, it is safe to share instances across threads. In
         multiprocessing scenarious, best practice is to create client
         instances after multiprocessing.Pool or multiprocessing.Process
         invokes os.fork().'

        So we run this on the first self.__call__ to make sure its in the
        subprocess created by dataloader.
        """
        from google.cloud import storage
        client = storage.Client()
        gs_bucket = client.get_bucket(bucket_name)
        return gs_bucket
