# Generated by Django 4.1 on 2022-08-17 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0017_sample_uploaded'),
    ]

    operations = [
        migrations.AddField(
            model_name='sample',
            name='generated',
            field=models.BooleanField(blank=True, default=False),
        ),
        migrations.AddField(
            model_name='sample',
            name='labelled',
            field=models.BooleanField(blank=True, default=False),
        ),
        migrations.AddField(
            model_name='sample',
            name='tested',
            field=models.BooleanField(blank=True, default=False),
        ),
    ]