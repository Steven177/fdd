# Generated by Django 3.2.13 on 2022-06-30 08:39

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0009_auto_20220628_2235'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='failure',
            name='samples',
        ),
        migrations.AddField(
            model_name='failure',
            name='sample',
            field=models.ForeignKey(blank=True, default=False, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.sample'),
        ),
    ]
