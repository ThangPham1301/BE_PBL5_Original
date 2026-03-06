from rest_framework.permissions import BasePermission, IsAuthenticated as DRFIsAuthenticated


IsAuthenticated = DRFIsAuthenticated


class IsAdmin(BasePermission):
    """Only admin users."""
    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role == 'admin'
        )


class IsAdminOrManager(BasePermission):
    """Admin or manager users."""
    def has_permission(self, request, view):
        return bool(
            request.user
            and request.user.is_authenticated
            and request.user.role in ('admin', 'manager')
        )


class IsOwnerOrAdmin(BasePermission):
    """
    Object-level permission: admin can access anything,
    others can only access their own objects.
    """
    def has_object_permission(self, request, view, obj):
        if request.user.role == 'admin':
            return True
        # Check if obj has employee → user chain
        if hasattr(obj, 'employee'):
            return obj.employee.user == request.user
        if hasattr(obj, 'user'):
            return obj.user == request.user
        return False


class IsManagerOfDepartment(BasePermission):
    """Manager can only access resources in their department."""
    def has_permission(self, request, view):
        if not request.user.is_authenticated:
            return False
        if request.user.role == 'admin':
            return True
        if request.user.role == 'manager':
            return True
        return False

    def has_object_permission(self, request, view, obj):
        if request.user.role == 'admin':
            return True
        if request.user.role == 'manager':
            try:
                manager_dept = request.user.employee.department
                if hasattr(obj, 'department'):
                    return obj.department == manager_dept
                if hasattr(obj, 'employee'):
                    return obj.employee.department == manager_dept
            except Exception:
                return False
        return False
