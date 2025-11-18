from django.contrib.gis.db.models import PointField
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        (
            "ppcx_app",
            "0013_add_image_rotation",
        ),
    ]

    operations = [
        migrations.RemoveField(
            model_name="collapse",
            name="centroid",
        ),
        migrations.AddField(
            model_name="collapse",
            name="centroid",
            field=PointField(null=True, blank=True, dim=2, srid=0),
        ),
    ]
