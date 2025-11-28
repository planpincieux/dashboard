from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("ppcx_app", "0027_remove_image_unique_camera_timestamp_and_more"),
    ]

    operations = [
        # Remove old fields
        migrations.RemoveField(
            model_name="dic",
            name="master_timestamp",
        ),
        migrations.RemoveField(
            model_name="dic",
            name="slave_timestamp",
        ),
        # Add them back as regular fields
        migrations.AddField(
            model_name="dic",
            name="master_timestamp",
            field=models.DateTimeField(null=True, blank=True, editable=False),
        ),
        migrations.AddField(
            model_name="dic",
            name="slave_timestamp",
            field=models.DateTimeField(null=True, blank=True, editable=False),
        ),
        # Populate existing data
        migrations.RunPython(
            code=lambda apps, schema_editor: populate_timestamps(apps),
            reverse_code=migrations.RunPython.noop,
        ),
        # Add indexes
        migrations.AddIndex(
            model_name="dic",
            index=models.Index(fields=["master_timestamp"], name="dic_master_ts_idx"),
        ),
        migrations.AddIndex(
            model_name="dic",
            index=models.Index(fields=["slave_timestamp"], name="dic_slave_ts_idx"),
        ),
    ]


def populate_timestamps(apps):
    """Populate timestamps from related images."""
    DIC = apps.get_model("ppcx_app", "DIC")
    for dic in DIC.objects.select_related("master_image", "slave_image").iterator():
        dic.master_timestamp = dic.master_image.datetime
        dic.slave_timestamp = dic.slave_image.datetime
        dic.save(update_fields=["master_timestamp", "slave_timestamp"])
