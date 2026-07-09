import gradio as gr
import joblib
import pandas as pd

# Cargar modelo y preprocesadores
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
selected_features = joblib.load("model/selected_features.pkl")
all_columns = joblib.load("model/all_columns.pkl")

def predecir(age, distance, num_companies, total_years, years_role,
             years_promotion, years_manager, job_involvement,
             overtime, job_role, marital_status):

    # Construir fila con TODAS las 43 columnas en ceros primero
    row = {col: 0 for col in all_columns}

    # Llenar los valores del usuario
    row['Age'] = age
    row['DistanceFromHome'] = distance
    row['NumCompaniesWorked'] = num_companies
    row['TotalWorkingYears'] = total_years
    row['YearsInCurrentRole'] = years_role
    row['YearsSinceLastPromotion'] = years_promotion
    row['YearsWithCurrManager'] = years_manager
    row['JobInvolvement'] = job_involvement
    row['OverTime'] = 1 if overtime == "Yes" else 0

    # Valores promedio para columnas que no preguntamos
    row['DailyRate'] = 800
    row['HourlyRate'] = 65
    row['MonthlyIncome'] = 5000
    row['MonthlyRate'] = 15000
    row['PercentSalaryHike'] = 14
    row['TrainingTimesLastYear'] = 3
    row['YearsAtCompany'] = total_years
    row['Education'] = 3
    row['EnvironmentSatisfaction'] = 3
    row['JobSatisfaction'] = 3
    row['RelationshipSatisfaction'] = 3
    row['WorkLifeBalance'] = 3
    row['JobLevel'] = 2
    row['StockOptionLevel'] = 1

    # One-hot encoding de JobRole
    job_role_col = f'JobRole_{job_role}'
    if job_role_col in row:
        row[job_role_col] = 1

    # One-hot encoding de MaritalStatus
    row[f'MaritalStatus_{marital_status}'] = 1

    # Respetar el orden de columnas usado al ajustar el scaler
    df = pd.DataFrame([row])[all_columns]

    # Escalar todas las columnas (el scaler fue ajustado sobre las 43 columnas)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=all_columns)

    # Seleccionar las 14 features del modelo
    df_final = df_scaled[selected_features]

    # Predecir
    pred = model.predict(df_final)[0]
    prob = model.predict_proba(df_final)[0][1]

    if pred == 1:
        return f"⚠️ Riesgo de salida ({prob:.0%} de probabilidad)"
    else:
        return f"✅ Empleado estable ({prob:.0%} de probabilidad de salida)"


interfaz = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Slider(18, 60, value=35, label="Edad"),
        gr.Slider(1, 29, value=5, label="Distancia al trabajo (km)"),
        gr.Slider(0, 9, value=2, label="Número de empresas anteriores"),
        gr.Slider(0, 40, value=10, label="Años de experiencia total"),
        gr.Slider(0, 18, value=3, label="Años en el rol actual"),
        gr.Slider(0, 15, value=2, label="Años desde última promoción"),
        gr.Slider(0, 17, value=3, label="Años con el mismo manager"),
        gr.Slider(1, 4, value=3, label="Nivel de involucramiento (1-4)"),
        gr.Radio(["Yes", "No"], value="No", label="¿Hace horas extra?"),
        gr.Dropdown(
            ["Laboratory Technician", "Research Director",
             "Sales Representative", "Other"],
            value="Other", label="Rol en la empresa"
        ),
        gr.Radio(["Single", "Married", "Divorced"],
                 value="Married", label="Estado civil"),
    ],
    outputs=gr.Text(label="Predicción"),
    title="🔍 Employee Attrition Predictor",
    description="Predice si un empleado tiene riesgo de dejar la empresa, basado en el dataset IBM HR Analytics.",
)

interfaz.launch(server_name="0.0.0.0", server_port=7860)