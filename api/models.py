from django.db import models
from django.contrib.auth.models import AbstractUser
from pgvector.django import VectorField
from django.utils import timezone

# -----------------------------
# 1. Custom User Model
# -----------------------------
class Organization(models.Model):
    name = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class User(AbstractUser):
    ROLE_CHOICES = [
        ('superadmin', 'Super Admin'),
        ('org_admin', 'Organization Admin'),
        ('member', 'Member'),
    ]
    organization = models.ForeignKey(Organization, on_delete=models.SET_NULL, null=True, blank=True, related_name='users')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='member')

    def __str__(self):
        return f"{self.username} ({self.role})"


# -----------------------------
# 2. Uploaded File
# -----------------------------
class UploadedFile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_files', null=True, blank=True)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name='uploaded_files', null=True, blank=True)
    file_name = models.CharField(max_length=255)
    file_size = models.FloatField()
    file_path = models.FileField(upload_to='uploads/%Y/%m/%d/')
    extracted_text = models.TextField(blank=True, null=True)
    upload_time = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.file_name} ({self.user.username if self.user else 'NoUser'})"


# -----------------------------
# 3. Embedded Document Chunks
# -----------------------------
class DocumentChunk(models.Model):
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE, related_name='chunks')
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, null=True, blank=True)
    content = models.TextField()
    embedding = VectorField(dimensions=768)
    chunk_index = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.uploaded_file.file_name}"


# -----------------------------
# 4. Chat / Query History
# -----------------------------
class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chats', null=True, blank=True)
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, null=True, blank=True)
    query = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Chat by {self.user.username if self.user else 'NoUser'} on {self.created_at.strftime('%Y-%m-%d %H:%M')}"
