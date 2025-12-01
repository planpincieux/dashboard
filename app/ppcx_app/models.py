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
    pixel_size_um = models.FloatField(
        null=True, blank=True, help_text="Physical pixel size in micrometers"
    )
    gsd = models.FloatField(
        null=True, blank=True, help_text="Average ground Sampling Distance in cm/pixel"
    )
    easting = models.FloatField(
        null=True, blank=True, help_text="Easting coordinate in meters (EPSG 32632)"
    )
    northing = models.FloatField(
        null=True, blank=True, help_text="Northing coordinate in meters (EPSG 32632)"
    )
    elevation = models.FloatField(
        null=True, blank=True, help_text="Elevation in meters (EPSG 32632)"
    )
    epsg_code = models.IntegerField(
        default=32632, help_text="EPSG code for the coordinate reference system"
    )
    installation_date = models.DateField(null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # PostGIS field for spatial representation
    location = gis_models.PointField(dim=3, srid=32632, null=True, blank=True)

    def save(self, *args, **kwargs):
        # Always include elevation (use 0 if not provided) for 3D point
        elevation = self.elevation if self.elevation is not None else 0
        # If easting and northing are provided, create a Point object
        if self.easting and self.northing:
            self.location = Point(
                self.easting,
                self.northing,
                elevation,
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
    datetime = models.DateTimeField(db_column="datetime")  # Primary column
    acquisition_timestamp = models.DateTimeField(
        null=True,
        blank=True,
        editable=False,
        db_comment="Deprecated: use datetime instead",
    )
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

    @property
    def date(self):
        """Extract date from datetime."""
        return self.datetime.date()

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
                fields=["camera", "datetime"],
                name="unique_camera_timestamp",
            )
        ]
        indexes = [
            models.Index(fields=["camera"]),
            models.Index(fields=["datetime"]),
        ]

    def __str__(self):
        return (
            f"{self.camera.camera_name} at {self.datetime.strftime('%Y-%m-%d %H:%M')}"
        )


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
        max_length=1024,
        null=True,
        blank=True,
        help_text="Path to the DIC results HDF5 file stored inside the server",
    )
    # Denormalized fields - synced automatically in save()
    master_timestamp = models.DateTimeField(null=True, blank=True, editable=False)
    slave_timestamp = models.DateTimeField(null=True, blank=True, editable=False)

    dt_hours = models.IntegerField(null=True, blank=True)
    software_used = models.CharField(max_length=100, null=True, blank=True)
    software_version = models.CharField(max_length=50, null=True, blank=True)
    processing_parameters = models.JSONField(null=True, blank=True)
    use_ensemble_correlation = models.BooleanField(
        default=False, help_text="Whether ensemble correlation was used"
    )
    ensemble_size = models.IntegerField(
        null=True, blank=True, help_text="Number of image pairs in the ensemble"
    )
    with_inversion = models.BooleanField(
        default=False,
        help_text="Whether time-series inversion was applied to refine DIC results",
    )
    # list of master/slave image id pairs used when ensemble correlation is active.
    # Format: [{"master": <image_id>, "slave": <image_id>}, ...]
    ensemble_image_pairs = models.JSONField(
        null=True,
        blank=True,
        help_text="List of {'master': image_id, 'slave': image_id} pairs used for ensemble correlation",
    )
    label = models.CharField(max_length=100, null=True, blank=True)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["reference_date"]),
            models.Index(fields=["master_timestamp"]),
            models.Index(fields=["slave_timestamp"]),
        ]

    def save(self, *args, **kwargs):
        # Auto-sync timestamps from related images
        if self.master_image:
            self.master_timestamp = self.master_image.datetime
        if self.slave_image:
            self.slave_timestamp = self.slave_image.datetime

        if not self.dt_hours and self.master_timestamp and self.slave_timestamp:
            time_diff = self.slave_timestamp - self.master_timestamp
            self.dt_hours = round(time_diff.total_seconds() / 3600)

        # Enforce logic: only keep ensemble_image_pairs if use_ensemble_correlation is true
        if not self.use_ensemble_correlation:
            self.ensemble_image_pairs = None
        else:
            # Basic validation of structure
            if self.ensemble_image_pairs:
                if not isinstance(self.ensemble_image_pairs, list):
                    raise ValueError("ensemble_image_pairs must be a list of dicts")
                cleaned = []
                for entry in self.ensemble_image_pairs:
                    if not isinstance(entry, dict):
                        raise ValueError("Each ensemble pair must be a dict")
                    if "master" not in entry or "slave" not in entry:
                        raise ValueError(
                            "Each pair must have 'master' and 'slave' keys"
                        )
                    if not isinstance(entry["master"], int) or not isinstance(
                        entry["slave"], int
                    ):
                        raise ValueError(
                            "'master' and 'slave' must be integer image IDs"
                        )
                    cleaned.append({"master": entry["master"], "slave": entry["slave"]})
                self.ensemble_image_pairs = cleaned

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

    def get_ensemble_images(self):
        """Return queryset of all unique images referenced in ensemble_image_pairs."""
        if not (self.use_ensemble_correlation and self.ensemble_image_pairs):
            return Image.objects.none()
        ids = set()
        for p in self.ensemble_image_pairs:
            ids.add(p["master"])
            ids.add(p["slave"])
        return Image.objects.filter(id__in=ids)


@receiver(post_delete, sender=DIC)
def delete_dic_file_on_signal(sender, instance, **kwargs):
    """Backup mechanism to delete files if the delete() method is bypassed."""
    if instance.result_file_path and Path(instance.result_file_path).exists():
        try:
            Path(instance.result_file_path).unlink()
        except (OSError, PermissionError) as e:
            logger.error(f"Signal error deleting file {instance.result_file_path}: {e}")


## ================ Collapses Model ================
class CollapseType(models.IntegerChoices):
    """Types of collapse events."""

    DISAGGREGATION = 1, "Disaggregation"
    SLAB = 2, "Slab"
    WATER_OUTBURST = 3, "Water outburst"


class Collapse(models.Model):
    # date = models.DateField(null=False)  # Date AFTER collapse (mandatory)
    image = models.ForeignKey(Image, on_delete=models.PROTECT, related_name="collapses")
    geom = gis_models.MultiPolygonField(
        null=True,
        blank=True,
        dim=2,
        srid=0,
        help_text="Original geometry in image coordinates (Y-axis down)",
    )
    geom_qgis = gis_models.MultiPolygonField(
        null=True,
        blank=True,
        dim=2,
        srid=0,
        editable=False,
        help_text="Y-inverted geometry for QGIS visualization (Y-axis up)",
    )
    type = models.IntegerField(
        choices=CollapseType.choices,
        null=True,
        blank=True,
        help_text="Type of collapse event",
    )
    area = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    centroid = gis_models.PointField(null=True, blank=True, dim=2, srid=0)
    notes = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Compute area and centroid if not provided
    # Note: geom_qgis is handled by database triggers
    def save(self, *args, **kwargs):
        if self.geom is not None:
            self.centroid = self.geom.centroid
            if self.area is None:
                self.area = self.geom.area

        super().save(*args, **kwargs)

    class Meta:
        db_table = "ppcx_app_collapse"
        managed = True  # Django still manages this table for queries

    def __str__(self) -> str:
        return f"Collapse {self.pk} (image={self.image})"
