# Generated by Django 3.2.13 on 2022-07-05 15:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0027_auto_20220705_1254'),
    ]

    operations = [
        migrations.AddField(
            model_name='match',
            name='exp_idx',
            field=models.IntegerField(blank=True, default=-1),
        ),
        migrations.AddField(
            model_name='match',
            name='pred_idx',
            field=models.IntegerField(blank=True, default=-1),
        ),
    ]