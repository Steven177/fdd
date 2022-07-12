from django.contrib import admin

# Register your models here.
from .models import Sample
from .models import Failure
from .models import Expectation
from .models import Model_Prediction
from .models import Match
from .models import Persona
from .models import Ai

admin.site.register(Sample)
admin.site.register(Failure)
admin.site.register(Expectation)
admin.site.register(Model_Prediction)
admin.site.register(Match)
admin.site.register(Persona)
admin.site.register(Ai)

