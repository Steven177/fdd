# Generated by Django 3.2.13 on 2022-06-27 00:36

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0004_auto_20220621_2130'),
    ]

    operations = [
        migrations.CreateModel(
            name='Expectation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(blank=True, max_length=100)),
                ('box', models.CharField(blank=True, max_length=100)),
                ('sample', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='fdd_app.sample')),
            ],
        ),
        migrations.DeleteModel(
            name='Test',
        ),
    ]