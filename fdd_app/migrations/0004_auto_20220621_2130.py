# Generated by Django 3.2.13 on 2022-06-21 21:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0003_alter_failure_failure_severity'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='failure',
            name='sample',
        ),
        migrations.AddField(
            model_name='failure',
            name='sample',
            field=models.ManyToManyField(blank=True, default=False, to='fdd_app.Sample'),
        ),
    ]