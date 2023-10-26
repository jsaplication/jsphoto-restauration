### JSPhoto Resturation
Reconstrução facil usando inteligencia artifical compativel com  CPU/GPU

Restaura suas fotos borradas. com esse incrivel algoritimo de inteligencia artifical que reconstroi seu rosto usando inteligencia artificial.

#### Importação da Biblioteca jwc-components.js
Importe via CDN ou NPM 
```Terminal
pip install -r requirements.txt
```


#### Rodar App com interface Gradio
Crie um localhost com interface gradio para testar a aplicação.

```Terminal
python app-gradio.py
```


Caso deseje usar em forma de API segue o codigo abaixo. lembrando para usar em forma de API primeiro teste o app em gradio para baixar os requesitos.

```Terminal
python rest-api.py
```

#### Como Solicitar. 
url: http://exemplo.com/reconstruir

paramentros: ---------------
```json
{
	"token": "api_key_abcd", // token definido no arquivo rest-api.py na linha 86
	"version": 1.4v //versão do modelo.
	"scale": 2, //escalatura da image exemplo qualidade quanto maior a escala mais qualidade a fodo fica mas tambem consome muito da sua gpu ou cpu
	"imagem": "image.png" // envie a image como multpart form data.
}
```



## Intergace Gradio

![Exemplo](imagens/captura.png)


## Foto Sem reconstrução facil
![Exemplo1](imagens/exemplo.jpeg)


## Foto com Reconstrução facil
![Exemplo2](imagens/outg.jpg)


Todo codigo foi baseado no [GFPGAN](https://github.com/TencentARC/GFPGAN)
Teste online: [Creditos Clem](https://huggingface.co/spaces/clem/Image_Face_Upscale_Restoration-GFPGAN)
Nosso Site [jsaplication.com.br](https://jsaplication.com.br)


Este projeto é licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.