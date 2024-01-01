from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from ..ml import xgboost_model


def index(request):
    temp = {}
    temp['SK_ID_CURR'] = 1
    temp['CODE_GENDER'] = 'F'
    temp['AMT_INCOME_TOTAL'] = 247500
    temp['AMT_CREDIT'] = 450000
    temp['AMT_ANNUITY'] = 27324
    temp['NAME_INCOME_TYPE'] = 'Working'
    temp['NAME_EDUCATION_TYPE'] = 'Higher education'
    temp['NAME_FAMILY_STATUS'] = 'Separated'
    temp['REGION_POPULATION_RELATIVE'] = 0.009175
    temp['DAYS_BIRTH'] = -13480
    temp['DAYS_EMPLOYED'] = -3009
    temp['DAYS_REGISTRATION'] = -4507
    temp['DAYS_ID_PUBLISH'] = -4323
    temp['OWN_CAR_AGE'] = ''
    temp['OCCUPATION_TYPE'] = 'Medicine staff'
    temp['WEEKDAY_APPR_PROCESS_START'] = 'TUESDAY'
    temp['HOUR_APPR_PROCESS_START'] = 15
    temp['ORGANIZATION_TYPE'] = 'Medicine'
    temp['EXT_SOURCE_1'] = ''
    temp['EXT_SOURCE_2'] = 0.745131
    temp['EXT_SOURCE_3'] = ''
    temp['LANDAREA_AVG'] = ''
    temp['APARTMENTS_MODE'] = ''
    temp['YEARS_BEGINEXPLUATATION_MEDI'] = ''
    temp['DAYS_LAST_PHONE_CHANGE'] = -970
    temp['FLAG_DOCUMENT_3'] = 1
    context = {'temp': temp}
    return render(request, 'index.html', context)

# короче так написать
# data:{
#                 sepal_length:$('#sepal_length').val(),
#                 sepal_width:$('#sepal_width').val(),
#                 petal_length:$('#petal_length').val(),
#                 petal_width:$('#petal_width').val(),
#                 csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val(),
#                 action: 'post'
#             },


def predictScore(request):
    if request.method == 'POST':
        temp = dict()
        temp['SK_ID_CURR'] = request.POST.get('skIdCurrVal')
        temp['CODE_GENDER'] = request.POST.get('codeGenderVal')
        temp['AMT_INCOME_TOTAL'] = request.POST.get('amtIncomeTotalVal')
        temp['AMT_CREDIT'] = request.POST.get('amtCreditVal')
        temp['AMT_ANNUITY'] = request.POST.get('amtAnnuityVal')
        temp['NAME_INCOME_TYPE'] = request.POST.get('incomeTypeVal')
        temp['NAME_EDUCATION_TYPE'] = request.POST.get('educationTypeVal')
        temp['NAME_FAMILY_STATUS'] = request.POST.get('familyStatusVal')
        temp['REGION_POPULATION_RELATIVE'] = request.POST.get('regionPopulationRelativeVal')
        temp['DAYS_BIRTH'] = request.POST.get('daysBirthVal')
        temp['DAYS_EMPLOYED'] = request.POST.get('daysEmployedVal')
        temp['DAYS_REGISTRATION'] = request.POST.get('daysRegistrationVal')
        temp['DAYS_ID_PUBLISH'] = request.POST.get('daysIdPublishVal')
        temp['OWN_CAR_AGE'] = request.POST.get('ownCarAgeVal')
        temp['OCCUPATION_TYPE'] = request.POST.get('occupationTypeVal')
        temp['WEEKDAY_APPR_PROCESS_START'] = request.POST.get('weekdayApprProcessStartVal')
        temp['HOUR_APPR_PROCESS_START'] = request.POST.get('hourApprProcessStartVal')
        temp['ORGANIZATION_TYPE'] = request.POST.get('organizationTypeVal')
        temp['EXT_SOURCE_1'] = request.POST.get('extSource1Val')
        temp['EXT_SOURCE_2'] = request.POST.get('extSource2Val')
        temp['EXT_SOURCE_3'] = request.POST.get('extSource3Val')
        temp['LANDAREA_AVG'] = request.POST.get('landareaAvgVal')
        temp['APARTMENTS_MODE'] = request.POST.get('apartmentModeVal')
        temp['YEARS_BEGINEXPLUATATION_MEDI'] = request.POST.get('yearsBeginExpluationMediVal')
        temp['DAYS_LAST_PHONE_CHANGE'] = request.POST.get('daysLastPhoneChangedVal')
        temp['FLAG_DOCUMENT_3'] = request.POST.get('flagDocument3Val')

        clf = xgboost_model.XGBoostClassifier()
        clf.compute_prediction(temp)
        scoreval = clf.compute_prediction(temp)

    context = {'scoreval': scoreval, 'temp': temp}
    return render(request, 'index.html', context)
