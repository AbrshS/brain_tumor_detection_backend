from rest_framework import serializers
from .models import CustomUser
from django.contrib.auth import authenticate

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = CustomUser
        fields = ['email', 'password', 'full_name', 'age', 'phone_number', 'hospital_name', 'hospital_department']

    def create(self, validated_data):
        user = CustomUser.objects.create_user(
            username=validated_data['email'],  # username required by AbstractUser
            email=validated_data['email'],
            password=validated_data['password'],
            full_name=validated_data.get('full_name', ''),
            age=validated_data.get('age'),
            phone_number=validated_data.get('phone_number', ''),
            hospital_name=validated_data.get('hospital_name', ''),
            hospital_department=validated_data.get('hospital_department', ''),
        )
        return user

class LoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

    def validate(self, data):
        user = authenticate(
            username=data['email'],
            password=data['password']
        )
        if not user:
            raise serializers.ValidationError("Invalid email or password.")
        data['user'] = user
        return data 