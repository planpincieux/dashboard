import logging
from enum import IntEnum
from pathlib import Path

from django.contrib.gis.db import models as gis_models
from django.contrib.gis.geos import Point
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models.signals import post_delete
from django.dispatch import receiver

logger = logging.getLogger("ppcx")  # Use the logger from the ppcx_app module

# ================ Camera and Calibration Models ================


class CameraModel(IntEnum):
    SIMPLE_PINHOLE = 0  # f, cx, cy
    PINHOLE = 1  # fx, fy, cx, cy
    SIMPLE_RADIAL = 2  # f, cx, cy, k1
    RADIAL = 3  # f, cx, cy, k1, k2
    OPENCV = 4  # fx, fy, cx, cy, k1, k2, p1, p2
    OPENCV_FISHEYE = 5  # fx, fy, cx, cy, k1, k2, k3, k4
    FULL_OPENCV = 6  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    FOV = 7  # fx, fy, cx, cy, omega
    SIMPLE_RADIAL_FISHEYE = 8  # f, cx, cy, k1
    RADIAL_FISHEYE = 9  # f, cx, cy, k1, k2
    THIN_PRISM_FISHEYE = 10  # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1

    @classmethod
    def choices(cls):
        return [(key.value, key.name) for key in cls]


class Camera(models.Model):
    """Information about each time-lapse camera."""

    camera_name = models.CharField(max_length=255)
    serial_number = models.CharField(max_length=100, unique=True, null=True, blank=True)
    model = models.CharField(max_length=100, null=True, blank=True)
    lens = models.CharField(max_length=100, null=True, blank=True)
    focal_length_mm = models.FloatField(null=True, blank=True)
    sensor_width_mm = models.FloatField(null=True, blank=True)
    sensor_height_mm = models.FloatField(null=True, blank=True)
    pixel_size_um = models.FloatField(null=True, blank=True)
    easting = models.FloatField(null=True, blank=True)
    northing = models.FloatField(null=True, blank=True)
    elevation = models.FloatField(null=True, blank=True)
    epsg_code = models.IntegerField(default=32632)
    installation_date = models.DateField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # PostGIS field for spatial representation
    location = gis_models.PointField(dim=3, srid=32632, null=True, blank=True)

    def save(self, *args, **kwargs):
        # If easting and northing are provided, create a Point object
        if self.easting and self.northing:
            self.location = Point(
                self.easting,
                self.northing,
                self.elevation if self.elevation else 0,
                srid=self.epsg_code,
            )

        # If location is provided, extract easting, northing, and elevation
        if self.easting is None and self.northing is None and self.location:
            self.easting = round(self.location.x, 3)
            self.northing = round(self.location.y, 3)
            self.elevation = (
                round(self.location.z, 3) if self.location.z is not None else None
            )

        super().save(*args, **kwargs)

    def __str__(self):
        return self.camera_name


class CameraCalibration(models.Model):
    """Camera interior and exterior orientation parameters."""

    camera = models.ForeignKey(
        Camera, on_delete=models.CASCADE, related_name="calibrations"
    )
    calibration_date = models.DateTimeField()
    is_active = models.BooleanField(default=True)
    colmap_model_id = models.IntegerField(choices=CameraModel.choices())
    colmap_model_name = models.CharField(max_length=50, null=True, blank=True)
    image_width_px = models.IntegerField()
    image_height_px = models.IntegerField()

    # Store intrinsic parameters as a float array to support all COLMAP models
    intrinsic_params = ArrayField(models.FloatField())

    # Extrinsic parameters
    rotation_quaternion = ArrayField(models.FloatField(), size=4, null=True, blank=True)
    translation_vector = ArrayField(models.FloatField(), size=3, null=True, blank=True)

    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["camera", "is_active"],
                condition=models.Q(is_active=True),
                name="unique_active_calibration",
            )
        ]

    def get_intrinsics_dict(self):
        """Convert the intrinsic parameters array to a dictionary with named parameters."""
        if self.colmap_model_id == CameraModel.SIMPLE_PINHOLE:
            return {
                "f": self.intrinsic_params[0],
                "cx": self.intrinsic_params[1],
                "cy": self.intrinsic_params[2],
            }
        elif self.colmap_model_id == CameraModel.PINHOLE:
            return {
                "fx": self.intrinsic_params[0],
                "fy": self.intrinsic_params[1],
                "cx": self.intrinsic_params[2],
                "cy": self.intrinsic_params[3],
            }
        elif self.colmap_model_id == CameraModel.SIMPLE_RADIAL:
            return {
                "f": self.intrinsic_params[0],
                "cx": self.intrinsic_params[1],
                "cy": self.intrinsic_params[2],
                "k1": self.intrinsic_params[3],
            }
        elif self.colmap_model_id == CameraModel.RADIAL:
            return {
                "f": self.intrinsic_params[0],
                "cx": self.intrinsic_params[1],
                "cy": self.intrinsic_params[2],
                "k1": self.intrinsic_params[3],
                "k2": self.intrinsic_params[4],
            }
        elif self.colmap_model_id == CameraModel.OPENCV:
            return {
                "fx": self.intrinsic_params[0],
                "fy": self.intrinsic_params[1],
                "cx": self.intrinsic_params[2],
                "cy": self.intrinsic_params[3],
                "k1": self.intrinsic_params[4],
                "k2": self.intrinsic_params[5],
                "p1": self.intrinsic_params[6],
                "p2": self.intrinsic_params[7],
            }
        # Add other models as needed
        return {"params": self.intrinsic_params}

    def __str__(self):
        return f"Calibration for {self.camera.camera_name} ({self.calibration_date.strftime('%Y-%m-%d')})"


# ================ Image Model ================


class Image(models.Model):
    """Metadata for each image acquired by the cameras."""

    camera = models.ForeignKey(Camera, on_delete=models.CASCADE, related_name="images")
    acquisition_timestamp = models.DateTimeField()
    file_path = models.CharField(max_length=1024, unique=True)
    width_px = models.IntegerField(null=True, blank=True)
    height_px = models.IntegerField(null=True, blank=True)
    exif_data = models.JSONField(null=True, blank=True)
    rotation = models.IntegerField(
        default=0, help_text="Rotation in degrees (0, 90, 180, 270)"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=100, null=True, blank=True)

    @property
    def file_name(self):
        """Extract filename from file_path."""
        return Path(self.file_path).name

    def save(self, *args, **kwargs):
        """Extract rotation from EXIF data if available."""
        if self.exif_data and isinstance(self.exif_data, dict):
            orientation = self.exif_data.get("Image Orientation", "")

            # Map EXIF orientation to rotation degrees
            orientation_map = {
                "Horizontal (normal)": 0,
                "Rotated 90 CW": 90,
                "Rotated 180": 180,
                "Rotated 90 CCW": 270,
                "Rotated 270 CW": 270,
            }
            self.rotation = orientation_map.get(orientation, 0)

        super().save(*args, **kwargs)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["camera", "acquisition_timestamp"],
                name="unique_camera_timestamp",
            )
        ]
        indexes = [
            models.Index(fields=["camera"]),
            models.Index(fields=["acquisition_timestamp"]),
        ]

    def __str__(self):
        return f"{self.camera.camera_name} at {self.acquisition_timestamp.strftime('%Y-%m-%d %H:%M')}"


# ================ DIC Models =============


class DIC(models.Model):
    """Information about DIC analysis between two images."""

    reference_date = models.DateField(null=True, blank=True)
    master_image = models.ForeignKey(
        Image, on_delete=models.CASCADE, related_name="master_dic"
    )
    slave_image = models.ForeignKey(
        Image, on_delete=models.CASCADE, related_name="slave_dic"
    )
    result_file_path = models.CharField(
        max_length=1024, null=True, blank=True
    )  # HDF5 file path
    master_timestamp = models.DateTimeField(null=True, blank=True)
    slave_timestamp = models.DateTimeField(null=True, blank=True)
    dt_hours = models.IntegerField(null=True, blank=True)
    label = models.CharField(max_length=100, null=True, blank=True)
    software_used = models.CharField(max_length=100, null=True, blank=True)
    software_version = models.CharField(max_length=50, null=True, blank=True)
    processing_parameters = models.JSONField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    use_ensemble_correlation = models.BooleanField(default=False)

    class Meta:
        constraints = [
            models.CheckConstraint(
                check=~models.Q(master_timestamp=models.F("slave_timestamp")),
                name="different_timestamps",
            ),
            models.UniqueConstraint(
                fields=["master_image", "slave_image", "dt_hours"],
                name="unique_image_pair_paths_dt_hours",
            ),
        ]
        indexes = [
            models.Index(fields=["reference_date"]),
            models.Index(fields=["master_timestamp"]),
            models.Index(fields=["slave_timestamp"]),
        ]

    def save(self, *args, **kwargs):
        if not self.dt_hours and self.master_timestamp and self.slave_timestamp:
            time_diff = self.slave_timestamp - self.master_timestamp
            self.dt_hours = round(time_diff.total_seconds() / 3600)
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Delete the h5 file when the DIC entry is deleted through Django."""
        file_path = self.result_file_path
        # Call the original delete method first
        result = super().delete(*args, **kwargs)

        # Then delete the associated file
        if file_path and Path(file_path).exists():
            try:
                Path(file_path).unlink()
                logger.info(f"Successfully deleted file: {file_path}")
            except (OSError, PermissionError) as e:
                logger.error(f"Error deleting file {file_path}: {e}")

        return result

    def __str__(self):
        return f"DIC: {self.master_timestamp} → {self.slave_timestamp}"


@receiver(post_delete, sender=DIC)
def delete_dic_file_on_signal(sender, instance, **kwargs):
    """Backup mechanism to delete files if the delete() method is bypassed."""
    if instance.result_file_path and Path(instance.result_file_path).exists():
        try:
            Path(instance.result_file_path).unlink()
        except (OSError, PermissionError) as e:
            logger.error(f"Signal error deleting file {instance.result_file_path}: {e}")


## ================ Collapses Model ================


class Collapse(models.Model):
    image = models.ForeignKey(Image, on_delete=models.PROTECT, related_name="collapses")
    geom = gis_models.PolygonField(null=True, blank=True, dim=2, srid=0)
    area = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    centroid = gis_models.PointField(
        null=True, blank=True, dim=2, srid=0
    )  # Change to PointField for centroid
    created_at = models.DateTimeField(auto_now_add=True)

    # Compute area if not provided
    def save(self, *args, **kwargs):
        if self.geom is not None:
            self.centroid = self.geom.centroid
            if self.area is None:
                self.area = self.geom.area

        super().save(*args, **kwargs)

    class Meta:
        db_table = "ppcx_app_collapse"

    def __str__(self) -> str:
        return f"Collapse {self.pk} (image={self.image})"
