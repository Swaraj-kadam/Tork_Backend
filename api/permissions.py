from rest_framework.permissions import BasePermission

class IsAdmin(BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'admin'

class IsOrganization(BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'organization'

class IsUser(BasePermission):
    def has_permission(self, request, view):
        return request.user.is_authenticated and request.user.role == 'user'
