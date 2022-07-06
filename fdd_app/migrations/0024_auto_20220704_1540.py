# Generated by Django 3.2.13 on 2022-07-04 15:40

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0023_auto_20220704_1536'),
    ]

    operations = [
        migrations.AlterField(
            model_name='match',
            name='expectation',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.expectation'),
        ),
        migrations.AlterField(
            model_name='match',
            name='model_prediction',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.model_prediction'),
        ),
    ]
