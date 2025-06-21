from django.urls import path
from .views import TumorAnalysisAPI, RegisterView, LoginView  # or PredictView if you kept the original name
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('predict/', TumorAnalysisAPI.as_view(), name='predict'),  # Updated to match your view class
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)