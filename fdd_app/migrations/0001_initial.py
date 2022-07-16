# Generated by Django 3.2.13 on 2022-07-16 16:01

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Ai',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=100)),
            ],
        ),
        migrations.CreateModel(
            name='Expectation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(blank=True, max_length=100)),
                ('xmin', models.IntegerField(blank=True, default=-1)),
                ('ymin', models.IntegerField(blank=True, default=-1)),
                ('xmax', models.IntegerField(blank=True, default=-1)),
                ('ymax', models.IntegerField(blank=True, default=-1)),
            ],
        ),
        migrations.CreateModel(
            name='Persona',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(blank=True, max_length=100)),
                ('personality', models.TextField(blank=True)),
                ('objectives_and_goals', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='Query',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('query', models.CharField(blank=True, max_length=200)),
            ],
        ),
        migrations.CreateModel(
            name='Sample',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('has_failure', models.BooleanField(blank=True, default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Scenario',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('context', models.CharField(blank=True, max_length=200)),
                ('persona', models.ForeignKey(blank=True, default=-1, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.persona')),
            ],
        ),
        migrations.CreateModel(
            name='Model_Prediction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('label', models.CharField(blank=True, max_length=100)),
                ('score', models.FloatField(blank=True, default=-1)),
                ('xmin', models.IntegerField(blank=True, default=-1)),
                ('ymin', models.IntegerField(blank=True, default=-1)),
                ('xmax', models.IntegerField(blank=True, default=-1)),
                ('ymax', models.IntegerField(blank=True, default=-1)),
                ('sample', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.sample')),
            ],
        ),
        migrations.CreateModel(
            name='Match',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('exp_idx', models.IntegerField(blank=True, default=-1)),
                ('pred_idx', models.IntegerField(blank=True, default=-1)),
                ('true_positive', models.BooleanField(blank=True, default=False)),
                ('failing_to_detect', models.BooleanField(blank=True, default=False)),
                ('false_detection', models.BooleanField(blank=True, default=False)),
                ('indistribution', models.BooleanField(blank=True, default=False)),
                ('missing_detection', models.BooleanField(blank=True, default=False)),
                ('unnecessary_detection', models.BooleanField(blank=True, default=False)),
                ('critical_quality_box', models.BooleanField(blank=True, default=False)),
                ('critical_quality_score', models.BooleanField(blank=True, default=False)),
                ('failure_severity', models.IntegerField(blank=True, default=0)),
                ('expectation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.expectation')),
                ('model_prediction', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.model_prediction')),
                ('sample', models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.sample')),
            ],
        ),
        migrations.AddField(
            model_name='expectation',
            name='sample',
            field=models.ForeignKey(blank=True, on_delete=django.db.models.deletion.CASCADE, to='fdd_app.sample'),
        ),
    ]
