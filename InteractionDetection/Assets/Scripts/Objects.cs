// store all the 20 labels that can be predicted 
// with this tiny yolov4 object detector, along with
// their descriptions
public class Objects{
    public static string[] labels = new[]
    {
        "Alimentatore", "Oscilloscopio", "Stazione di saldatura", "Avvitatore elettrico",
        "Cacciavite", "Pinza", "Punta del saldatore", "Sonda oscilloscopio",
        "Scheda a bassa tensione", "Scheda ad alta tensione", "register", "Batteria avvitatore",
        "Area di lavoro", "Base saldatore", "Presa", "left red button",
        "left green button", "right red button", "right green button", "hand"
    };

    public static string[] descriptions = new []
    {
        "L'alimentatore di corrente è uno strumento che rifornisce di energia i dispositivi a lui collegati.", 
        "L'oscilloscopio è uno strumento di misura elettronico che consente di visualizzare l'andamento dei segnali elettrici nel tempo ed effettuare misurazioni.",
        "Una stazione di saldatura è un dispositivo di saldatura multiuso progettato per la saldatura di componenti elettronici.",
        "Un avvitatore elettrico è uno strumento atto ad avvitare le viti e praticare fori di vario genere utilizzando l'elettricità.",
        "Il cacciavite è un attrezzo utilizzato per avvitare o svitare viti costituito da un'impugnatura e un tipo di punta.",
        "La pinza è un utensile utilizzato per afferrare, stringere, unire e tagliare gli oggetti su cui si lavora.",
        "La punta del saldatore serve per saldare componenti tramite fusione, grazie al calore fornito dalla stazione di saldatura.",
        "La sonda dell'oscilloscopio permette di prelevare il segnale da esaminare e di trasferirlo allo strumento.",
        "La scheda a bassa tensione è un componente hardware ovvero un circuito stampato deputato ad un certo tipo di elaborazione elettronica.",
        "La scheda ad alta tensione è un componente hardware ovvero un circuito stampato deputato ad un certo tipo di elaborazione elettronica.",
        "",
        "La batteria dell'avvitatore elettrico è un componente che permette di immagazzinare energia e fornirla allo strumento.",
        "Un'area di lavoro è un luogo in cui vengono effettuate delle lavorazioni di vario tipo, come saldatura etc.",
        "La base del saldatore è un componente della stazione di saldatura e serve per poter poggiare il saldatore in tutta sicurezza.",
        "Una presa serve per poter ricaricare dispositivi elettronici mediante corrente elettrica alternata.",
        "",
        "",
        "",
        "",
        ""
    };

    public static int index(string label){
        int i = 0;
        for(; i < labels.Length; i++){
            if(labels[i] == label)
                break;
        }
        return i;
    }
}

