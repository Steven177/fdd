# Generated by Django 3.2.13 on 2022-07-16 16:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='scenario',
            name='name',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]
