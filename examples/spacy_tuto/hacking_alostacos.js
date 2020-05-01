//reaccion
function capturaTecla(e)
{
	var habilitaEspacio;

	habilitaEspacio = true;
	nSalidas++;
	if( !puedePulsar )
	{
		clearTimeout(idListos);
		clearTimeout(idYa);
		document.getElementById('avisoComienzo').innerHTML = 'Te has escapado';
		document.getElementById('pantallaJuego').innerHTML += '<p class=\'sinMargen\'>Te has escapado.</p>';
		nNulas++;
	}
	else
	{
        timestampPulsado = new Date().getTime();
        function getRandomArbitrary(min, max) {
            var r = 0;
            var v = 2;
            for(var i = v; i > 0; i --){
                r += Number.parseFloat((Math.random() * (max - min) + min).toFixed(3))
            }
            return (r / v).toFixed(3);
          }

          tiempoReaccion = getRandomArbitrary(0.14, 0.22)
		// tiempoReaccion = ((timestampPulsado - timestampYa) / 1000 );
		if( tiempoReaccion < 0.15 )
			puntosAcumulados += 3;
		else if( tiempoReaccion < 0.2 )
			puntosAcumulados += 2;
		else if( tiempoReaccion <= 0.25 )
			puntosAcumulados++;

		txtMensaje = 'Reacci&oacute;n: ' + tiempoReaccion;
		document.getElementById('totalPuntosAcumulados').innerHTML = 'Puntos acumulados: ' + puntosAcumulados;
		document.getElementById('avisoComienzo').innerHTML = txtMensaje;
		document.getElementById('pantallaJuego').innerHTML += '<p class=\'sinMargen\'>' + txtMensaje + '</p>';
	}

	document.getElementById('mensaje').style.display = 'block';
	continuar = false;

	if( nNulas ===  2 )
	{
		document.getElementById('mensaje').innerHTML = '<p class="sinMargen">Te has escapado dos veces.</p>';
		document.getElementById('mensaje').innerHTML += '<p class="sinMargen">Quedas descalificado y pierdes los puntos que llevaras acumulados.</p>';
		document.getElementById('mensaje').innerHTML += '<p>Pulsa la barra espaciadora para entrenar de nuevo</p>';
	}
	else if( nSalidas === 5 )
	{
		if( puntosAcumulados < 1 )
		{
			document.getElementById('mensaje').innerHTML = '<p class="sinMargen">No has logrado ningúna bonificaci&oacute;n.</p>';
			document.getElementById('mensaje').innerHTML += '<p>Pulsa la barra espaciadora para entrenar de nuevo</p>';
		}
		else
		{
			habilitaEspacio = false;
			document.getElementById('resultados').value = document.getElementById('pantallaJuego').innerHTML;
			document.getElementById('puntos').value = puntosAcumulados;
			document.getElementById('frmJuego').submit();
		}
	}
	else
	{
		document.getElementById('mensaje').innerHTML = 'Pulsa la barra espaciadora para continuar.';
		continuar = true;
	}

	if( habilitaEspacio )
		document.onkeypress = capturaEspacio;
	else
		document.onkeypress = false;
}



// velocidad
function iniciaJuego()
{
	numeroAciertos = 0;
	modoJuego = document.getElementById('modoJuego1').checked ? 1 : 2;

	cObjetivo = document.getElementById('capa');

	if( modoJuego === 1 )
	{
 		if( cObjetivo.addEventListener )
 			cObjetivo.addEventListener('mousedown', funUnBoton, true);
  	else
			cObjetivo.attachEvent('onmousedown', funUnBoton);
	}
	else
	{
 		if( cObjetivo.addEventListener )
 			cObjetivo.addEventListener('mousedown', funDosBotones, true);
  	else
			cObjetivo.attachEvent('onmousedown', funDosBotones);
	}

	document.oncontextmenu = function() {return false};

	document.getElementById('avisoComienzo').style.display = 'none';
	document.getElementById('capa').style.display = 'block';

	actualiza();
	setTimeout('finJuego()', 2500);
}

function finJuego()
{
	document.getElementById('capa').style.display = 'none';
    minimoAciertos = modoJuego + 6;
    
    function getRandomArbitrary(min, max) {
        var r = 0;
        var v = 2;
        for(var i = v; i > 0; i --){
            r += Number.parseFloat((Math.random() * (max - min) + min))
        }
        return (r / v);
      }

    let res = Math.round(getRandomArbitrary(1,4))

	if( true )
	{
		document.getElementById('puntos').value = res
		document.getElementById('frmJuego').submit();
	}
	else
	{
		if( modoJuego === 1 )
		{
 			if( cObjetivo.addEventListener )
 				cObjetivo.removeEventListener('mousedown', funUnBoton, true);
	  	else
				cObjetivo.detachEvent('onmousedown', funUnBoton);
		}
		else
		{
 			if( cObjetivo.addEventListener )
 				cObjetivo.removeEventListener('mousedown', funDosBotones, true);
	  	else
				cObjetivo.detachEvent('onmousedown', funDosBotones);
		}
		document.onmousedown = function() {return true};
		document.oncontextmenu = function() {return true};

		document.getElementById('pantallaJuego').style.display = 'none';
		document.getElementById('instrucciones').style.display = 'block';
		document.getElementById('mensaje').style.display = 'block';
		document.getElementById('mensaje').innerHTML = '¡Has sido muy lento! ¡No has ganado ningún punto de entrenamiento!';
	}
}


function generaCarretera()
{
	var longCarreteraTotal, tramo1, tramo2;

	longCarreteraTotal = NUMERO_TRAMOS * LARGO_CARRETERA;
	carretera1 = new Array(longCarreteraTotal);
	carretera2 = new Array(longCarreteraTotal);
	posCarretera1 = new Array(longCarreteraTotal);
	posCarretera2 = new Array(longCarreteraTotal);

	for( var i = 0; i < NUMERO_TRAMOS -1 ; i++)
	{
		var limite = (i + 1) * LARGO_CARRETERA;

		tramo1 = TRAMOS_CARRETERA[0];
		tramo2 = TRAMOS_CARRETERA[0];

		for(var x = i * LARGO_CARRETERA, j = 0; x < limite; x++, j++)
		{
			carretera1[x] = '<div class="carretera" style="margin-left:' + tramo1[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
			carretera2[x] = '<div class="carretera" style="margin-left:' + tramo2[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
			posCarretera1[x] = tramo1[j];
			posCarretera2[x] = tramo2[j];
		}
	}

	// Creo el primer y último tramo que será recto.
	tramo1 = TRAMOS_CARRETERA[0];
	for( var i = longCarreteraTotal - LARGO_CARRETERA, j = 0; i < longCarreteraTotal; i++, j++ )
	{
		carretera1[i] = '<div class="carretera" style="margin-left:' + tramo1[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
		carretera2[i] = '<div class="carretera" style="margin-left:' + tramo1[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
		posCarretera1[i] = tramo1[j];
		posCarretera2[i] = tramo1[j];
		carretera1[j] = '<div class="carretera" style="margin-left:' + tramo1[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
		carretera2[j] = '<div class="carretera" style="margin-left:' + tramo1[j] + 'px; height:' + ALTO_LINEA + 'px; width:' + ANCHO_CARRETERA + 'px;"></div>';
		posCarretera1[j] = tramo1[j];
		posCarretera2[j] = tramo1[j];
	}
}