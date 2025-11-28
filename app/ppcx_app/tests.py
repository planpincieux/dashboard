import json
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from django.test import Client, TestCase
from django.utils import timezone

import ppcx_app.views as views
from ppcx_app.models import DIC, Camera, Image


class UploadDicH5Tests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Minimal camera
        cls.camera = Camera.objects.create(camera_name="PPCX_Wide")

        # Create two images on different dates (make timezone-aware)
        dt_master = timezone.make_aware(datetime(2021, 8, 3, 12, 0))
        dt_slave = timezone.make_aware(datetime(2021, 8, 5, 12, 0))

        # Some projects use both acquisition_timestamp and datetime; set both if present
        img_common = dict(camera=cls.camera, rotation=0)
        cls.master_image = Image.objects.create(
            file_path="/tmp/master_image.jpg",
            acquisition_timestamp=dt_master,
            datetime=dt_master if hasattr(Image, "datetime") else dt_master,
            **{k: v for k, v in img_common.items() if hasattr(Image, k)},
        )
        cls.slave_image = Image.objects.create(
            file_path="/tmp/slave_image.jpg",
            acquisition_timestamp=dt_slave,
            datetime=dt_slave if hasattr(Image, "datetime") else dt_slave,
            **{k: v for k, v in img_common.items() if hasattr(Image, k)},
        )

    def setUp(self):
        # Patch H5 output dir to a temp folder for each test
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        views.H5_FILES_DIR = Path(self.tmpdir.name)
        self.client = Client()

    def _post(self, payload: dict):
        return self.client.post(
            "/API/dic/upload/",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_upload_by_dates_minimal(self):
        payload = {
            "master_date": "2021-08-03",
            "slave_date": "2021-08-05",
            "dic_data": {
                "points": [[0, 0], [10, 20]],
                "magnitudes": [0.5, 1.2],
            },
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 200, r.content)
        data = r.json()
        self.assertIn("dic_id", data)
        self.assertIn("h5_path", data)

        # H5 file exists and contains required datasets; no vectors/max_magnitude
        h5_path = Path(data["h5_path"])
        self.assertTrue(h5_path.exists(), "H5 not written")
        with h5py.File(h5_path, "r") as f:
            self.assertIn("points", f.keys())
            self.assertIn("magnitudes", f.keys())
            self.assertNotIn("vectors", f.keys())
            self.assertNotIn("max_magnitude", f.keys())
            pts = f["points"][()]
            mags = f["magnitudes"][()]
            np.testing.assert_array_equal(
                pts, np.array([[0, 0], [10, 20]], dtype=np.int32)
            )
            np.testing.assert_allclose(mags, np.array([0.5, 1.2], dtype=np.float32))

        # Filename pattern includes id and stems
        dic_obj = DIC.objects.get(id=data["dic_id"])
        self.assertIn(f"{dic_obj.pk:08d}_", h5_path.name)
        self.assertIn(Path(self.master_image.file_path).stem, h5_path.name)
        self.assertIn(Path(self.slave_image.file_path).stem, h5_path.name)

    def test_upload_by_ids_full_options(self):
        payload = {
            "master_image_id": self.master_image.id,
            "slave_image_id": self.slave_image.id,
            "reference_date": "2021-08-05",
            "software_used": "pylamma",
            "software_version": "2.1.0",
            "with_inversion": True,
            "use_ensemble_correlation": False,
            "ensemble_size": 3,
            "processing_parameters": {"window": 32},
            "label": "unit-test",
            "notes": "all optional fields",
            "dt_hours": 48,
            "dic_data": {
                "points": [[1, 2], [3, 4]],
                "vectors": [[0.1, 0.2], [0.3, 0.4]],
                "magnitudes": [0.5, 0.6],
                "max_magnitude": 0.6,
            },
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 200, r.content)
        dic_id = r.json()["dic_id"]
        dic = DIC.objects.get(id=dic_id)
        self.assertEqual(dic.software_used, "pylamma")
        if hasattr(dic, "software_version"):
            self.assertEqual(dic.software_version, "2.1.0")
        self.assertTrue(dic.with_inversion)
        if hasattr(dic, "use_ensemble_correlation"):
            self.assertFalse(dic.use_ensemble_correlation)
        if hasattr(dic, "ensemble_size"):
            self.assertEqual(dic.ensemble_size, 3)
        if hasattr(dic, "processing_parameters"):
            self.assertEqual(dic.processing_parameters.get("window"), 32)
        self.assertEqual(dic.label, "unit-test")
        self.assertEqual(dic.notes, "all optional fields")
        self.assertEqual(dic.dt_hours, 48)

        # H5 has vectors and max_magnitude
        with h5py.File(dic.result_file_path, "r") as f:
            self.assertIn("vectors", f.keys())
            self.assertIn("max_magnitude", f.keys())

    def test_type_coercion_and_defaults(self):
        payload = {
            "master_date": "2021-08-03",
            "slave_date": "2021-08-05",
            # Provide values as strings to test coercion
            "with_inversion": "true",
            "ensemble_size": "5",
            "dt_hours": "24",
            # Omit software_used -> default "pylamma"
            "dic_data": {
                "points": [[5, 6]],
                "magnitudes": [1.0],
            },
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 200, r.content)
        dic = DIC.objects.get(id=r.json()["dic_id"])
        self.assertTrue(dic.with_inversion)
        if hasattr(dic, "ensemble_size"):
            self.assertEqual(dic.ensemble_size, 5)
        self.assertEqual(dic.dt_hours, 24)
        self.assertEqual(dic.software_used, "pylamma")

    def test_invalid_date_returns_404(self):
        payload = {
            "master_date": "2021/08/03",  # invalid
            "slave_date": "2021-08-05",
            "dic_data": {"points": [[0, 0]], "magnitudes": [0.1]},
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 404)

    def test_missing_dic_data_returns_400(self):
        payload = {
            "master_date": "2021-08-03",
            "slave_date": "2021-08-05",
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 400)

    def test_custom_dic_fields_are_persisted(self):
        payload = {
            "master_date": "2021-08-03",
            "slave_date": "2021-08-05",
            "dic_data": {
                "points": [[0, 0]],
                "magnitudes": [0.1],
                "quality": [0.9],  # stored as dataset
                "comment": "unit-test",  # stored as HDF5 attribute
            },
        }
        r = self._post(payload)
        self.assertEqual(r.status_code, 200, r.content)
        dic = DIC.objects.get(id=r.json()["dic_id"])
        with h5py.File(dic.result_file_path, "r") as f:
            self.assertIn("quality", f.keys())
            np.testing.assert_allclose(f["quality"][()], np.array([0.9]))
            self.assertEqual(f.attrs.get("comment"), "unit-test")
