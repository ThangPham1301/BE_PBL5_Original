from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.token_blacklist.models import OutstandingToken, BlacklistedToken

from .serializers import LoginSerializer, UserSerializer


@api_view(['POST'])
@permission_classes([AllowAny])
def login_view(request):
    serializer = LoginSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    user = serializer.validated_data['user']
    refresh = RefreshToken.for_user(user)
    return Response({
        'success': True,
        'data': {
            'access': str(refresh.access_token),
            'refresh': str(refresh),
            'user': UserSerializer(user).data,
        },
        'message': 'Đăng nhập thành công.',
    })


@api_view(['POST'])
@permission_classes([AllowAny])
def refresh_view(request):
    refresh_token = request.data.get('refresh')
    if not refresh_token:
        return Response({
            'success': False,
            'data': None,
            'message': 'Thiếu mã làm mới phiên đăng nhập.',
        }, status=status.HTTP_400_BAD_REQUEST)
    try:
        refresh = RefreshToken(refresh_token)
        return Response({
            'success': True,
            'data': {
                'access': str(refresh.access_token),
            },
            'message': 'Đã làm mới phiên đăng nhập thành công.',
        })
    except Exception:
        return Response({
            'success': False,
            'data': None,
            'message': 'Mã làm mới phiên đăng nhập không hợp lệ.',
        }, status=status.HTTP_401_UNAUTHORIZED)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_view(request):
    try:
        refresh_token = request.data.get('refresh')
        if refresh_token:
            token = RefreshToken(refresh_token)
            token.blacklist()
        return Response({
            'success': True,
            'data': None,
            'message': 'Đăng xuất thành công.',
        })
    except Exception:
        return Response({
            'success': False,
            'data': None,
            'message': 'Mã phiên đăng nhập không hợp lệ.',
        }, status=status.HTTP_400_BAD_REQUEST)
