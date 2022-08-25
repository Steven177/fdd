# Generated by Django 4.1 on 2022-08-24 07:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fdd_app', '0020_match_outofdistribution'),
    ]

    operations = [
        migrations.CreateModel(
            name='Suggestion',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=500)),
                ('match', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.match')),
            ],
        ),
    ]