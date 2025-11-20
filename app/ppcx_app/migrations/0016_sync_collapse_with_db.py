from django.contrib.gis.db import models as gis_models
from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("ppcx_app", "0015_reset_centroid_field"),
    ]

    operations = [
        # Change geom from Polygon to MultiPolygon
        migrations.AlterField(
            model_name="collapse",
            name="geom",
            field=gis_models.MultiPolygonField(
                blank=True,
                dim=2,
                help_text="Original geometry in image coordinates (Y-axis down)",
                null=True,
                srid=0,
            ),
        ),
        # Add geom_qgis field
        migrations.AddField(
            model_name="collapse",
            name="geom_qgis",
            field=gis_models.MultiPolygonField(
                blank=True,
                dim=2,
                editable=False,
                help_text="Y-inverted geometry for QGIS visualization (Y-axis up)",
                null=True,
                srid=0,
            ),
        ),
    ]
