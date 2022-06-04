from django.db import models

# Create your models here.
class Google(models.Model):
    Date = models.CharField(max_length=50, primary_key=True)
    High = models.FloatField()
    Low = models.FloatField()
    Open = models.FloatField()
    Close = models.FloatField()
    Volume = models.BigIntegerField()
    Adj_Close = models.FloatField()

    def __str__(self):
        return self.Date.strftime('%Y-%m-%d')


class Twitter(models.Model):
    Date = models.CharField(max_length=50, primary_key=True)
    High = models.FloatField()
    Low = models.FloatField()
    Open = models.FloatField()
    Close = models.FloatField()
    Volume = models.BigIntegerField()
    Adj_Close = models.FloatField()

    def __str__(self):
        return self.Date.strftime('%Y-%m-%d')


class Apple(models.Model):
    Date = models.CharField(max_length=50, primary_key=True)
    High = models.FloatField()
    Low = models.FloatField()
    Open = models.FloatField()
    Close = models.FloatField()
    Volume = models.BigIntegerField()
    Adj_Close = models.FloatField()

    def __str__(self):
        return self.Date.strftime('%Y-%m-%d')


class Microsoft(models.Model):
    Date = models.CharField(max_length=50, primary_key=True)
    High = models.FloatField()
    Low = models.FloatField()
    Open = models.FloatField()
    Close = models.FloatField()
    Volume = models.BigIntegerField()
    Adj_Close = models.FloatField()

    def __str__(self):
        return self.Date.strftime('%Y-%m-%d')