import calendar
from io import BytesIO
from datetime import date

from django.http import HttpResponse
from django.db.models import Count, Q
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side, PatternFill

from apps.accounts.permissions import IsAuthenticated, IsAdminOrManager
from apps.employees.models import Employee, Department
from apps.attendance.models import AttendanceLog
from .serializers import MonthlyReportQuerySerializer, ExportQuerySerializer


class ReportViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminOrManager]

    @action(detail=False, methods=['get'], url_path='monthly')
    def monthly(self, request):
        """
        Monthly attendance report.
        GET /api/reports/monthly/?month=3&year=2026&department_id=1
        """
        serializer = MonthlyReportQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data

        month = params['month']
        year = params['year']
        department_id = params.get('department_id')

        employees = Employee.objects.filter(is_active=True).select_related('user', 'department')

        # Scope by role
        user = request.user
        if user.is_manager:
            try:
                manager_employee = user.employee
                employees = employees.filter(department=manager_employee.department)
            except Employee.DoesNotExist:
                employees = employees.none()
        elif department_id:
            employees = employees.filter(department_id=department_id)

        _, days_in_month = calendar.monthrange(year, month)
        start = date(year, month, 1)
        end = date(year, month, days_in_month)

        report_data = []
        for emp in employees:
            logs = AttendanceLog.objects.filter(
                employee=emp, date__gte=start, date__lte=end
            )
            summary = logs.aggregate(
                present=Count('id', filter=Q(status='present')),
                late=Count('id', filter=Q(status='late')),
                absent=Count('id', filter=Q(status='absent')),
                leave=Count('id', filter=Q(status='leave')),
            )
            report_data.append({
                'employee_id': emp.employee_id,
                'employee_name': emp.user.get_full_name() or emp.user.username,
                'department': emp.department.name if emp.department else None,
                'total_days': days_in_month,
                'present': summary['present'],
                'late': summary['late'],
                'absent': summary['absent'],
                'leave': summary['leave'],
                'working_days': summary['present'] + summary['late'],
            })

        return Response({
            'success': True,
            'data': {
                'month': month,
                'year': year,
                'employees': report_data,
            },
            'message': '',
        })

    @action(detail=False, methods=['get'], url_path='export')
    def export(self, request):
        """
        Export monthly report to Excel.
        GET /api/reports/export/?month=3&year=2026&department_id=1
        """
        serializer = ExportQuerySerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data

        month = params['month']
        year = params['year']
        department_id = params.get('department_id')

        employees = Employee.objects.filter(is_active=True).select_related('user', 'department')

        user = request.user
        if user.is_manager:
            try:
                manager_employee = user.employee
                employees = employees.filter(department=manager_employee.department)
            except Employee.DoesNotExist:
                employees = employees.none()
        elif department_id:
            employees = employees.filter(department_id=department_id)

        _, days_in_month = calendar.monthrange(year, month)
        start = date(year, month, 1)
        end = date(year, month, days_in_month)

        # Build workbook
        wb = Workbook()
        ws = wb.active
        ws.title = f'Báo cáo T{month}/{year}'

        # Styles
        header_font = Font(bold=True, size=12)
        title_font = Font(bold=True, size=14)
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin'),
        )
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        header_font_white = Font(bold=True, size=11, color='FFFFFF')

        # Title
        ws.merge_cells('A1:H1')
        ws['A1'] = f'BÁO CÁO CHẤM CÔNG THÁNG {month}/{year}'
        ws['A1'].font = title_font
        ws['A1'].alignment = Alignment(horizontal='center')

        # Headers
        headers = ['STT', 'Mã NV', 'Họ Tên', 'Phòng Ban', 'Có Mặt', 'Đi Trễ', 'Vắng', 'Nghỉ Phép']
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=3, column=col_idx, value=header)
            cell.font = header_font_white
            cell.fill = header_fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center')

        # Data rows
        row_num = 4
        for idx, emp in enumerate(employees, 1):
            logs = AttendanceLog.objects.filter(
                employee=emp, date__gte=start, date__lte=end
            )
            summary = logs.aggregate(
                present=Count('id', filter=Q(status='present')),
                late=Count('id', filter=Q(status='late')),
                absent=Count('id', filter=Q(status='absent')),
                leave=Count('id', filter=Q(status='leave')),
            )
            row_data = [
                idx,
                emp.employee_id,
                emp.user.get_full_name() or emp.user.username,
                emp.department.name if emp.department else '',
                summary['present'],
                summary['late'],
                summary['absent'],
                summary['leave'],
            ]
            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row_num, column=col_idx, value=value)
                cell.border = thin_border
                if col_idx >= 5:
                    cell.alignment = Alignment(horizontal='center')
            row_num += 1

        # Column widths
        col_widths = [6, 12, 25, 20, 10, 10, 10, 12]
        for i, width in enumerate(col_widths, 1):
            ws.column_dimensions[chr(64 + i)].width = width

        # Write to buffer
        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)

        response = HttpResponse(
            buffer.getvalue(),
            content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
        response['Content-Disposition'] = f'attachment; filename="attendance_report_{month}_{year}.xlsx"'
        return response
