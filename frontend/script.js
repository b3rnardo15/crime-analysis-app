// Variáveis globais
let dadosCasos = [];
let graficoRosca = null;
let graficoDistribuicao = null;
let graficoModelo = null;
let graficoTemporal = null;
let graficoBoxplot = null;
let graficoEspacial = null;
let graficoClustering = null;
let graficoRegressao = null;
let graficoTipos = null;

const gradiente = [
  "#40516c",
  "#4a5d7c",
  "#53698c",
  "#5d759c",
  "#6b82a7",
  "#7b90b1",
  "#8b9dba",
];

const coresClusters = [
  "#FF5733",
  "#33FF57",
  "#3357FF",
  "#F033FF",
  "#FF3333",
];

document.addEventListener("DOMContentLoaded", () => {
  // Definir datas padrão (últimos 30 dias)
  const fim = new Date();
  const inicio = new Date();
  inicio.setDate(inicio.getDate() - 30);

  document.getElementById("dataInicio").valueAsDate = inicio;
  document.getElementById("dataFim").valueAsDate = fim;
});

async function carregarDados() {
    try {
        console.log("Iniciando carregamento de dados...");
        const res = await fetch("http://127.0.0.1:5000/api/victims");
        if (!res.ok) {
            throw new Error(`HTTP error! status: ${res.status}`);
        }
        dadosCasos = await res.json();
        console.log("Dados carregados:", dadosCasos);
        
        if (dadosCasos.length === 0) {
            console.warn("Nenhuma vítima foi carregada da API");
            alert("Não há vítimas disponíveis no banco de dados.");
            return;
        }
        
        atualizarGraficos();
        
        // Carregar dados para os gráficos de machine learning
        inicializarGraficoModelo();
        carregarDadosClustering();
        carregarDadosRegressao();
        carregarDadosBoxplot();
    } catch (erro) {
        console.error("Erro ao carregar dados:", erro);
        alert("Erro ao carregar os dados: " + erro.message);
    }
}

function filtrarPorData(casos) {
  const inicio = document.getElementById("dataInicio").value;
  const fim = document.getElementById("dataFim").value;

  return casos.filter((caso) => {
    // Usar locationDate ou createdAt como referência de data
    const dataRef = caso.locationDate || caso.createdAt;
    if (!dataRef) return false;
    
    const data = new Date(dataRef);
    const dataInicio = inicio ? new Date(inicio) : null;
    const dataFim = fim ? new Date(fim) : null;
    return (!dataInicio || data >= dataInicio) && (!dataFim || data <= dataFim);
  });
}
function processarDadosTipos(casos) {
  // Usar identificationType como tipo de caso para as vítimas
  return contarOcorrencias(casos, 'identificationType');
}
function atualizarGraficos() {
    try {
        const dadosFiltrados = filtrarPorData(dadosCasos);
        
        // 1. Gráfico de Status dos Casos (Rosca)
        const variavelRosca = document.getElementById("variavelRosca").value;
        atualizarGraficoRosca(dadosFiltrados, variavelRosca);
        
        // 2. Gráfico Temporal de Casos por Mês
        const dadosTemporais = processarDadosTemporais(dadosFiltrados);
        atualizarGraficoTemporal(dadosTemporais);
        
        // 3. Gráfico de Tipos de Casos
        const dadosTipos = processarDadosTipos(dadosFiltrados);
        atualizarGraficoTipos(dadosTipos);
        
        // 4. Gráfico de Características das Vítimas
        const dadosVitimas = processarDadosVitimas(dadosFiltrados);
        atualizarGraficoVitimas(dadosVitimas);
        
        console.log("Todos os gráficos foram atualizados com sucesso");
    } catch (erro) {
        console.error("Erro ao atualizar gráficos:", erro);
    }
}

function processarDadosVitimas(casos) {
    return {
        identificacao: contarValores(casos, 'identificationType'),
        genero: contarValores(casos, 'gender'),
        nacionalidade: contarValores(casos, 'nationality'),
        condicaoCorpo: contarValores(casos, 'bodyCondition')
    };
}

function contarValores(array, campo) {
    return array.reduce((acc, item) => {
        const valor = item[campo] || 'não especificado';
        acc[valor] = (acc[valor] || 0) + 1;
        return acc;
    }, {});
}

function contarOcorrencias(array, campo) {
    return array.reduce((acc, item) => {
        const valor = item[campo] || 'não especificado';
        acc[valor] = (acc[valor] || 0) + 1;
        return acc;
    }, {});
}

function atualizarGraficoRosca(dadosFiltrados, variavel = "tipo_do_caso") {
  const contagem = contarOcorrencias(dadosFiltrados, variavel);
  const labels = Object.keys(contagem);
  const valores = Object.values(contagem);
  const cores = gradiente.slice(0, labels.length);

  const ctx = document.createElement("canvas");
  document.getElementById("graficoRosca").innerHTML = "";
  document.getElementById("graficoRosca").appendChild(ctx);

  if (graficoRosca) graficoRosca.destroy();

  graficoRosca = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: labels,
      datasets: [
        {
          data: valores,
          backgroundColor: cores,
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        title: {
          display: true,
          text: 'Frequência Relativa dos Casos'
        }
      },
    },
  });
}

function atualizarGraficoDistribuicao(dadosFiltrados) {
  const idades = dadosFiltrados
    .map((c) => c.vitima?.idade)
    .filter((i) => typeof i === "number" && !isNaN(i) && i > 0);

  const max = Math.max(...idades, 100);
  const bins = [];
  const labels = [];

  for (let i = 1; i <= max; i += 10) {
    labels.push(`${i}-${i + 9}`);
    bins.push(0);
  }

  idades.forEach((idade) => {
    const index = Math.floor((idade - 1) / 10);
    if (index >= 0 && index < bins.length) {
      bins[index]++;
    }
  });

  const ctx = document.createElement("canvas");
  document.getElementById("graficoDistribuicao").innerHTML = "";
  document.getElementById("graficoDistribuicao").appendChild(ctx);

  if (graficoDistribuicao) graficoDistribuicao.destroy();

  graficoDistribuicao = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Número de Vítimas",
          data: bins,
          backgroundColor: "#5d759c",
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
        },
      },
      plugins: {
        title: {
          display: true,
          text: 'Distribuição de Idades'
        }
      }
    },
  });
}

async function inicializarGraficoModelo() {
  try {
    const res = await fetch("http://localhost:5000/api/modelo/coeficientes");
    const data = await res.json();

    const processedData = {};
    Object.keys(data).forEach((key) => {
      processedData[key] = Number(data[key]);
    });

    const sortedEntries = Object.entries(processedData).sort(
      (a, b) => Math.abs(b[1]) - Math.abs(a[1])
    );

    const labels = sortedEntries.map(([key]) => key);
    const valores = sortedEntries.map(([, value]) => value);

    const ctx = document.createElement("canvas");
    document.getElementById("graficoModelo").innerHTML = "";
    document.getElementById("graficoModelo").appendChild(ctx);

    if (graficoModelo) graficoModelo.destroy();

    graficoModelo = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Importância",
            data: valores,
            backgroundColor: "#5d759c",
            borderWidth: 1,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: {
            display: false,
          },
          title: {
            display: true,
            text: 'Importância das Features no Modelo'
          }
        },
      },
    });
  } catch (error) {
    console.error("Erro ao carregar coeficientes:", error);
  }
}

// Novas funções para os gráficos adicionais

function processarDadosTemporais(casos) {
    const dadosPorMes = {};
    casos.forEach(caso => {
        // Usar locationDate ou createdAt como referência de data
        const dataRef = caso.locationDate || caso.createdAt;
        if (!dataRef) return;
        
        const data = new Date(dataRef);
        const mesAno = `${data.getFullYear()}-${String(data.getMonth() + 1).padStart(2, '0')}`;
        dadosPorMes[mesAno] = (dadosPorMes[mesAno] || 0) + 1;
    });
    return dadosPorMes;
}

function atualizarGraficoTemporal(dadosTemporais) {
    const labels = Object.keys(dadosTemporais).sort();
    const valores = labels.map(label => dadosTemporais[label]);

    const ctx = document.createElement('canvas');
    document.getElementById('graficoTemporal').innerHTML = '';
    document.getElementById('graficoTemporal').appendChild(ctx);

    if (graficoTemporal) graficoTemporal.destroy();

    graficoTemporal = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Número de Casos',
                data: valores,
                borderColor: '#5d759c',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Distribuição Temporal de Casos'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

function atualizarGraficoTipos(dadosTipos) {
    const labels = Object.keys(dadosTipos);
    const valores = Object.values(dadosTipos);

    const ctx = document.createElement('canvas');
    document.getElementById('graficoTipos').innerHTML = '';
    document.getElementById('graficoTipos').appendChild(ctx);

    if (graficoTipos) graficoTipos.destroy();

    graficoTipos = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Número de Casos',
                data: valores,
                backgroundColor: gradiente[0],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Distribuição por Tipo de Caso'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

async function carregarDadosBoxplot() {
  try {
    const res = await fetch("http://localhost:5000/api/casos/estatisticas/boxplot");
    const data = await res.json();
    
    const tipos = Object.keys(data);
    const datasets = [];
    
    tipos.forEach((tipo, index) => {
      const boxplotData = data[tipo];
      datasets.push({
        label: tipo,
        backgroundColor: gradiente[index % gradiente.length] + '80',
        borderColor: gradiente[index % gradiente.length],
        borderWidth: 1,
        outliers: boxplotData.outliers,
        itemRadius: 3,
        itemStyle: 'circle',
        itemBackgroundColor: gradiente[index % gradiente.length],
        data: [{
          min: boxplotData.min,
          q1: boxplotData.q1,
          median: boxplotData.median,
          q3: boxplotData.q3,
          max: boxplotData.max,
          outliers: boxplotData.outliers
        }]
      });
    });
    
    const ctx = document.createElement("canvas");
    document.getElementById("graficoBoxplot").innerHTML = "";
    document.getElementById("graficoBoxplot").appendChild(ctx);
    
    if (graficoBoxplot) graficoBoxplot.destroy();
    
    // Nota: Boxplot requer o plugin Chart.js BoxPlot
    // Aqui estamos simulando com um gráfico de barras para simplificar
    graficoBoxplot = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: tipos,
        datasets: [{
          label: 'Mediana',
          data: tipos.map(tipo => data[tipo].median),
          backgroundColor: gradiente.map(cor => cor + '80'),
          borderColor: gradiente,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Idade'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Comparação de Idades por Tipo de Caso (Mediana)'
          },
          tooltip: {
            callbacks: {
              afterLabel: function(context) {
                const tipo = context.label;
                const stats = data[tipo];
                return [
                  `Min: ${stats.min}`,
                  `Q1: ${stats.q1.toFixed(1)}`,
                  `Mediana: ${stats.median.toFixed(1)}`,
                  `Q3: ${stats.q3.toFixed(1)}`,
                  `Max: ${stats.max}`
                ];
              }
            }
          }
        }
      }
    });
  } catch (error) {
    console.error("Erro ao carregar dados para boxplot:", error);
  }
}

async function carregarDadosClustering() {
  try {
    const res = await fetch("http://localhost:5000/api/ml/clustering");
    const data = await res.json();
    
    // Preparar dados para o gráfico de dispersão
    const datasets = data.map((cluster, index) => ({
      label: `Cluster ${cluster.cluster_id} (${cluster.tamanho} casos)`,
      data: cluster.pontos.map(p => ({ x: p.x, y: new Date(p.y * 1000).toISOString().split('T')[0], tipo: p.tipo })),
      backgroundColor: coresClusters[index % coresClusters.length],
      pointRadius: 8,
      pointHoverRadius: 12
    }));
    
    const ctx = document.createElement("canvas");
    document.getElementById("graficoClustering").innerHTML = "";
    document.getElementById("graficoClustering").appendChild(ctx);
    
    if (graficoClustering) graficoClustering.destroy();
    
    graficoClustering = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: datasets
      },
      options: {
        responsive: true,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Idade da Vítima'
            }
          },
          y: {
            type: 'time',
            time: {
              unit: 'day'
            },
            title: {
              display: true,
              text: 'Data do Caso'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Clusterização dos Casos'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                const point = context.raw;
                return [`Idade: ${point.x}`, `Data: ${point.y}`, `Tipo: ${point.tipo}`];
              }
            }
          }
        }
      }
    });
    
    // Adicionar informações sobre os clusters
    const infoDiv = document.createElement('div');
    infoDiv.className = 'cluster-info';
    infoDiv.innerHTML = '<h3>Informações dos Clusters</h3>';
    
    data.forEach((cluster, index) => {
      const clusterDiv = document.createElement('div');
      clusterDiv.style.marginBottom = '15px';
      clusterDiv.style.padding = '10px';
      clusterDiv.style.backgroundColor = '#f5f5f5';
      clusterDiv.style.borderLeft = `4px solid ${coresClusters[index % coresClusters.length]}`;
      
      let tiposHtml = '';
      Object.entries(cluster.tipos_caso).forEach(([tipo, count]) => {
        tiposHtml += `<li>${tipo}: ${count} casos</li>`;
      });
      
      clusterDiv.innerHTML = `
        <h4>Cluster ${cluster.cluster_id}</h4>
        <p>Tamanho: ${cluster.tamanho} casos</p>
        <p>Idade média: ${cluster.idade_media.toFixed(1)} anos</p>
        <p>Tipos de casos:</p>
        <ul>${tiposHtml}</ul>
      `;
      
      infoDiv.appendChild(clusterDiv);
    });
    
    document.getElementById("graficoClustering").appendChild(infoDiv);
    
  } catch (error) {
    console.error("Erro ao carregar dados de clustering:", error);
  }
}

async function carregarDadosRegressao() {
  try {
    const res = await fetch("http://localhost:5000/api/ml/regressao");
    const data = await res.json();
    
    // Gráfico de dispersão para valores reais vs. previstos
    const ctx = document.createElement("canvas");
    document.getElementById("graficoRegressao").innerHTML = "";
    document.getElementById("graficoRegressao").appendChild(ctx);
    
    if (graficoRegressao) graficoRegressao.destroy();
    
    graficoRegressao = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Valores Reais vs. Previstos',
          data: data.actual.map((actual, i) => ({ x: actual, y: data.predicted[i] })),
          backgroundColor: '#5d759c',
          pointRadius: 8
        }, {
          label: 'Linha Ideal (y=x)',
          data: (() => {
            const min = Math.min(...data.actual, ...data.predicted);
            const max = Math.max(...data.actual, ...data.predicted);
            return [{ x: min, y: min }, { x: max, y: max }];
          })(),
          type: 'line',
          borderColor: '#FF5733',
          borderWidth: 2,
          pointRadius: 0,
          fill: false
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            title: {
              display: true,
              text: 'Idade Real'
            }
          },
          y: {
            title: {
              display: true,
              text: 'Idade Prevista'
            }
          }
        },
        plugins: {
          title: {
            display: true,
            text: `Modelo de Regressão (R² = ${data.r2.toFixed(3)})`
          }
        }
      }
    });
    
    // Adicionar informações sobre os coeficientes
    const infoDiv = document.createElement('div');
    infoDiv.className = 'regression-info';
    infoDiv.innerHTML = '<h3>Coeficientes do Modelo</h3>';
    
    const coefsList = document.createElement('ul');
    Object.entries(data.coeficientes)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .forEach(([feature, coef]) => {
        const li = document.createElement('li');
        li.textContent = `${feature}: ${coef.toFixed(3)}`;
        coefsList.appendChild(li);
      });
    
    infoDiv.appendChild(coefsList);
    document.getElementById("graficoRegressao").appendChild(infoDiv);
    
  } catch (error) {
    console.error("Erro ao carregar dados de regressão:", error);
  }
}

// Adiciona os event listeners para os campos de data e variável
document
  .getElementById("dataInicio")
  .addEventListener("change", atualizarGraficos);
document
  .getElementById("dataFim")
  .addEventListener("change", atualizarGraficos);
document
  .getElementById("variavelRosca")
  .addEventListener("change", atualizarGraficos);

// Inicia o carregamento dos dados
carregarDados();


// Adicione esta variável junto com as outras variáveis globais no início do arquivo
let graficoVitimas = null;

function atualizarGraficoVitimas(dadosVitimas) {
    const ctx = document.createElement('canvas');
    document.getElementById('graficoVitimas').innerHTML = '';
    document.getElementById('graficoVitimas').appendChild(ctx);

    if (graficoVitimas) graficoVitimas.destroy();

    const categorias = Object.keys(dadosVitimas);
    const datasets = categorias.map((categoria, index) => {
        const dados = dadosVitimas[categoria];
        return {
            label: categoria,
            data: Object.values(dados),
            backgroundColor: gradiente[index % gradiente.length],
            borderWidth: 0
        };
    });

    graficoVitimas = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: Object.keys(dadosVitimas[categorias[0]] || {}),
            datasets: datasets
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Características das Vítimas'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
