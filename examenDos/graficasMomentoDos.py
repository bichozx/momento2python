import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------
# 1️⃣ Generar dataset simulado
# -----------------------------------

np.random.seed(42)

n = 30  # cantidad de estudiantes

nombres = [f"Estudiante_{i}" for i in range(1, n + 1)]
edades = np.random.randint(18, 35, n)
semestres = np.random.choice([1, 3, 5, 7, 9], n)
habilidades = np.random.choice([
    "Python, Django, SQL",
    "JavaScript, React, CSS",
    "Java, Spring Boot, SQL",
    "HTML, CSS, JS",
    "Node.js, Express, MongoDB",
    "React, TypeScript, Redux"
], n)
certificados = np.random.choice([
    "AWS", "Azure", "Google Cloud", "Scrum", "AWS", "SQL"
], n)
proyectos = np.random.choice([
    "E-commerce", "Portfolio", "API REST", "Dashboard", "Portfolio"
], n)
resumenes = np.random.choice([
    "Desarrollador con pasión por crear soluciones eficientes.",
    "Interesado en desarrollo web moderno.",
    "Experiencia en backend con APIs escalables.",
    "Enfocado en frontend con React y UI/UX.",
    "Estudiante con interés en arquitectura de software."
], n)
completitud = np.random.choice([True, True, True, False], n)  # algunos incompletos

df = pd.DataFrame({
    "nombre": nombres,
    "edad": edades,
    "semestre": semestres,
    "habilidades": habilidades,
    "certificados": certificados,
    "proyectos": proyectos,
    "resumen": resumenes,
    "completo": completitud
})

print("=== Dataset simulado ===")
print(df.head(), "\n")

# -----------------------------------
# 2️⃣ Clasificar perfiles automáticamente
# -----------------------------------

def clasificar_perfil(hab):
    if "React" in hab or "JS" in hab:
        return "Frontend"
    elif "Python" in hab or "Django" in hab:
        return "Backend"
    elif "Java" in hab or "Spring" in hab:
        return "Backend"
    elif "Node" in hab or "Express" in hab:
        return "Full Stack"
    else:
        return "Otro"

df["perfil"] = df["habilidades"].apply(clasificar_perfil)

# -----------------------------------
# 3️⃣ Comparar longitud promedio del resumen por semestre
# -----------------------------------

df["longitud_resumen"] = df["resumen"].apply(len)

promedio_por_semestre = df.groupby("semestre")["longitud_resumen"].mean().reset_index()

plt.figure(figsize=(7,4))
sns.barplot(data=promedio_por_semestre, x="semestre", y="longitud_resumen", palette="mako")
plt.title("Longitud promedio del resumen profesional por semestre")
plt.show()

# -----------------------------------
# 4️⃣ Detectar duplicados en certificados o proyectos
# -----------------------------------

duplicados_cert = df[df.duplicated("certificados", keep=False)]
duplicados_proy = df[df.duplicated("proyectos", keep=False)]

print("=== Certificados duplicados ===")
print(duplicados_cert[["nombre", "certificados"]].sort_values("certificados"), "\n")

print("=== Proyectos duplicados ===")
print(duplicados_proy[["nombre", "proyectos"]].sort_values("proyectos"), "\n")

# -----------------------------------
# 5️⃣ Agrupar por tecnologías dominantes
# -----------------------------------

conteo_perfil = df["perfil"].value_counts().reset_index()
conteo_perfil.columns = ["perfil", "cantidad"]

plt.figure(figsize=(6,4))
sns.barplot(data=conteo_perfil, x="perfil", y="cantidad", palette="Set2")
plt.title("Distribución de estudiantes por perfil tecnológico")
plt.show()

# -----------------------------------
# 6️⃣ Filtrar hojas de vida incompletas
# -----------------------------------

incompletos = df[df["completo"] == False]
print("=== Hojas de vida incompletas ===")
print(incompletos[["nombre", "completo"]], "\n")

# -----------------------------------
# 7️⃣ Mostrar resumen general
# -----------------------------------

plt.figure(figsize=(8,5))
sns.countplot(data=df, x="semestre", hue="perfil", palette="coolwarm")
plt.title("Perfiles por semestre académico")
plt.show()
