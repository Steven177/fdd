# Generated by Django 4.1 on 2022-08-24 10:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0021_suggestion'),
    ]

    operations = [
        migrations.AddField(
            model_name='suggestion',
            name='challenge',
            field=models.BooleanField(blank=True, default=False),
        ),
        migrations.AddField(
            model_name='suggestion',
            name='challenge_again',
            field=models.BooleanField(blank=True, default=False),
        ),
        migrations.AddField(
            model_name='suggestion',
            name='guide',
            field=models.BooleanField(blank=True, default=False),
        ),
    ]
