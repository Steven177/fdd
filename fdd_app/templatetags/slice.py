from django import template
register = template.Library()


@register.filter
def slice(string):
  return string[0:2]
