# Generated by Django 3.2.13 on 2022-07-12 11:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0033_persona'),
    ]

    operations = [
        migrations.CreateModel(
            name='Ai',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=100)),
            ],
        ),
    ]
