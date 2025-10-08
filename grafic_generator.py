import plotly.graph_objects as go
import pandas as pd

# Create the data
data = [
    {"Level":"1","Model":"Random Forest","Metric":"Accuracy","Value":0.8477},
    {"Level":"1","Model":"Random Forest","Metric":"F1","Value":0.8146},
    {"Level":"1","Model":"Logistic Regression","Metric":"Accuracy","Value":0.8451},
    {"Level":"1","Model":"Logistic Regression","Metric":"F1","Value":0.8208},
    {"Level":"1","Model":"SVM","Metric":"Accuracy","Value":0.8923},
    {"Level":"1","Model":"SVM","Metric":"F1","Value":0.8667},
    {"Level":"1","Model":"Naive Bayes","Metric":"Accuracy","Value":0.8387},
    {"Level":"1","Model":"Naive Bayes","Metric":"F1","Value":0.8264},
    {"Level":"2","Model":"Random Forest","Metric":"Accuracy","Value":0.7285},
    {"Level":"2","Model":"Random Forest","Metric":"F1","Value":0.6961},
    {"Level":"2","Model":"Logistic Regression","Metric":"Accuracy","Value":0.7231},
    {"Level":"2","Model":"Logistic Regression","Metric":"F1","Value":0.7186},
    {"Level":"2","Model":"SVM","Metric":"Accuracy","Value":0.8441},
    {"Level":"2","Model":"SVM","Metric":"F1","Value":0.8386},
    {"Level":"2","Model":"Naive Bayes","Metric":"Accuracy","Value":0.8138},
    {"Level":"2","Model":"Naive Bayes","Metric":"F1","Value":0.8126},
    {"Level":"3","Model":"Random Forest","Metric":"Accuracy","Value":0.662},
    {"Level":"3","Model":"Random Forest","Metric":"F1","Value":0.6247},
    {"Level":"3","Model":"Logistic Regression","Metric":"Accuracy","Value":0.6485},
    {"Level":"3","Model":"Logistic Regression","Metric":"F1","Value":0.6336},
    {"Level":"3","Model":"SVM","Metric":"Accuracy","Value":0.7843},
    {"Level":"3","Model":"SVM","Metric":"F1","Value":0.7774},
    {"Level":"3","Model":"Naive Bayes","Metric":"Accuracy","Value":0.6996},
    {"Level":"3","Model":"Naive Bayes","Metric":"F1","Value":0.6653}
]

df = pd.DataFrame(data)

# Abbreviate model names to meet 15-char limit
model_abbrev = {
    "Random Forest": "Random Forest",
    "Logistic Regression": "Log Regress", 
    "SVM": "SVM",
    "Naive Bayes": "Naive Bayes"
}

df['Model_Short'] = df['Model'].map(model_abbrev)

# Create the figure
fig = go.Figure()

# Brand colors for the 6 combinations (3 levels × 2 metrics)
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F', '#D2BA4C', '#B4413C']

# Get unique combinations and models
combinations = []
for level in ['1', '2', '3']:
    for metric in ['Accuracy', 'F1']:
        combinations.append(f"L{level}-{metric}")

models = ['Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes']

# Add a trace for each level-metric combination
for i, (level, metric) in enumerate([('1','Accuracy'), ('1','F1'), ('2','Accuracy'), ('2','F1'), ('3','Accuracy'), ('3','F1')]):
    subset = df[(df['Level'] == level) & (df['Metric'] == metric)]
    
    fig.add_trace(go.Bar(
        name=f"L{level}-{metric}",
        x=subset['Model_Short'],
        y=subset['Value'],
        marker_color=colors[i % len(colors)],
        offsetgroup=i,
        legendgroup=f"L{level}-{metric}"
    ))

fig.update_layout(
    title="Model Performance by Level & Metric",
    xaxis_title="Models",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1]),
    barmode='group',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

fig.update_traces(cliponaxis=False)

# Save as both PNG and SVG
fig.write_image("model_performance.png")