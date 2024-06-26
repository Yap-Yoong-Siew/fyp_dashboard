/*

CUSTOMIZE THE FOLLOWING TO GET YOUR OWN EXPERT ADVISOR UP AND RUNNING

EvaluateEntry : To insert your custom entry signal

EvaluateExit : To insert your custom exit signal

ExecuteTrailingStop : To insert your trailing stop rules

StopLossPriceCalculate : To set your custom Stop Loss value

TakeProfitPriceCalculate : To set your custom Take Profit value

*/

//-PROPERTIES-//
//Properties help the software look better when you load it in MT4
//Provide more information and details
//This is what you see in the About tab when you load an Indicator or an Expert Advisor
#property link          "https://github.com/Yap-Yoong-Siew/Prop"
#property version       "1.00"

#property copyright     "Yap Yoong Siew 2024"
#property description   "This codebase is for Final Year Project" 
#property description   " "
#property description   "WARNING : You use this software at your own risk."
#property description   "The creator of these plugins cannot be held responsible for any damage or loss."
#property description   " "
#property description   "Please contact the developer for any questions at s2122194@siswa.um.edu.my"
//You can add an icon for when the EA loads on chart but it's not necessary
//The commented line below is an example of icon, icon must be in the MQL4/Files folder and have a ico extension


//-INCLUDES-//
//Include allows to import code from another file
//In the following instance the file has to be placed in the MQL4/Include Folder
#include <postgremql4.mqh>
#include <MQLTA ErrorHandling.mqh>

string g_db_ip_setting = "localhost";
string g_db_port_setting = "5432";
string g_db_user_setting = "postgres";
string g_db_password_setting = "1234";
string g_db_name_setting = "postgres";
//-COMMENTS-//
//This is a single line comment and I do it placing // at the start of the comment, this text is ignored when compiling

/*
This is a multi line comment
it starts with /* and it finishes with the * and / like below
*/


//-ENUMERATIVE VARIABLES-//
//Enumerative variables are useful to associate numerical values to easy to remember strings
//It is similar to constants but also helps if the variable is set from the input page of the EA
//The text after the // is what you see in the input paramenters when the EA loads
//It is good practice to place all the enumberative at the start

//Enumerative for the hour of the day
enum ENUM_HOUR{
   h00=00,     //00:00
   h01=01,     //01:00
   h02=02,     //02:00
   h03=03,     //03:00
   h04=04,     //04:00
   h05=05,     //05:00
   h06=06,     //06:00
   h07=07,     //07:00
   h08=08,     //08:00
   h09=09,     //09:00
   h10=10,     //10:00
   h11=11,     //11:00
   h12=12,     //12:00
   h13=13,     //13:00
   h14=14,     //14:00
   h15=15,     //15:00
   h16=16,     //16:00
   h17=17,     //17:00
   h18=18,     //18:00
   h19=19,     //19:00
   h20=20,     //20:00
   h21=21,     //21:00
   h22=22,     //22:00
   h23=23,     //23:00
};

//Enumerative for the entry signal value
enum ENUM_SIGNAL_ENTRY{
   SIGNAL_ENTRY_NEUTRAL=0,    //SIGNAL ENTRY NEUTRAL
   SIGNAL_ENTRY_BUY=1,        //SIGNAL ENTRY BUY
   SIGNAL_ENTRY_SELL=-1,      //SIGNAL ENTRY SELL
};

//Enumerative for the exit signal value
enum ENUM_SIGNAL_EXIT{
   SIGNAL_EXIT_NEUTRAL=0,     //SIGNAL EXIT NEUTRAL
   SIGNAL_EXIT_BUY=1,         //SIGNAL EXIT BUY
   SIGNAL_EXIT_SELL=-1,       //SIGNAL EXIT SELL
   SIGNAL_EXIT_ALL=2,         //SIGNAL EXIT ALL
};

//Enumerative for the allowed trading direction
enum ENUM_TRADING_ALLOW_DIRECTION{
   TRADING_ALLOW_BOTH=0,      //ALLOW BOTH BUY AND SELL
   TRADING_ALLOW_BUY=1,       //ALLOW BUY ONLY
   TRADING_ALLOW_SELL=-1,     //ALLOW SELL ONLY
};

//Enumerative for the base used for risk calculation
enum ENUM_RISK_BASE{
   RISK_BASE_EQUITY=1,        //EQUITY
   RISK_BASE_BALANCE=2,       //BALANCE
   RISK_BASE_FREEMARGIN=3,    //FREE MARGIN
};

//Enumerative for the default risk size
enum ENUM_RISK_DEFAULT_SIZE{
   RISK_DEFAULT_FIXED=1,      //FIXED SIZE
   RISK_DEFAULT_AUTO=2,       //AUTOMATIC SIZE BASED ON RISK
};

//Enumerative for the Stop Loss mode
enum ENUM_MODE_SL{
   SL_FIXED=0,                //FIXED STOP LOSS
   SL_AUTO=1,                 //AUTOMATIC STOP LOSS
};

//Enumerative for the Take Profit Mode
enum ENUM_MODE_TP{
   TP_FIXED=0,                //FIXED TAKE PROFIT
   TP_AUTO=1,                 //AUTOMATIC TAKE PROFIT
};

//Enumerative for the stop loss calculation
enum ENUM_MODE_SL_BY{
   SL_BY_POINTS=0,            //STOP LOSS PASSED IN POINTS
   SL_BY_PRICE=1,             //STOP LOSS PASSED BY PRICE
};


//-INPUT PARAMETERS-//
//The input parameters are the ones that can be set by the user when launching the EA
//If you place a comment following the input variable this will be shown as description of the field

//This is where you should include the input parameters for your entry and exit signals
input string Comment_strategy="==========";                          //Entry And Exit Settings
//Add in this section the parameters for the indicators used in your entry and exit

string rsi_setting="==========";                               //RSI Settings
input int rsi_fast_period = 14;                                   //RSI Fast Period
input int start_hour = 0;                                       // start time (hour)
input int end_hour = 23;                                       // end time (hour)
input double lot_size = 0.01;
input double tp_percent = 1.5;
input double sl_percent = 1.0;



//General input parameters
input string Comment_0="==========";                                 //Risk Management Settings
input ENUM_RISK_DEFAULT_SIZE RiskDefaultSize=RISK_DEFAULT_AUTO;      //Position Size Mode
input double DefaultLotSize=1;                                       //Position Size (if fixed or if no stop loss defined)
input ENUM_RISK_BASE RiskBase=RISK_BASE_BALANCE;                     //Risk Base
input int MaxRiskPerTrade=2;                                         //Percentage To Risk Each Trade
input double MinLotSize=0.01;                                        //Minimum Position Size Allowed
input double MaxLotSize=100;                                         //Maximum Position Size Allowed

input string Comment_1="==========";                                 //Trading Hours Settings
input bool UseTradingHours=false;                                    //Limit Trading Hours
input ENUM_HOUR TradingHourStart=h07;                                //Trading Start Hour (Broker Server Hour)
input ENUM_HOUR TradingHourEnd=h19;                                  //Trading End Hour (Broker Server Hour)

input string Comment_2="==========";                                 //Stop Loss And Take Profit Settings
input ENUM_MODE_SL StopLossMode=SL_FIXED;                            //Stop Loss Mode
input int DefaultStopLoss=0;                                         //Default Stop Loss In Points (0=No Stop Loss)
input int MinStopLoss=0;                                             //Minimum Allowed Stop Loss In Points
input int MaxStopLoss=5000;                                          //Maximum Allowed Stop Loss In Points
input ENUM_MODE_TP TakeProfitMode=TP_FIXED;                          //Take Profit Mode
input int DefaultTakeProfit=0;                                       //Default Take Profit In Points (0=No Take Profit)
input int MinTakeProfit=0;                                           //Minimum Allowed Take Profit In Points
input int MaxTakeProfit=5000;                                        //Maximum Allowed Take Profit In Points

input string Comment_3="==========";                                 //Trailing Stop Settings
input bool UseTrailingStop=false;                                    //Use Trailing Stop

input string Comment_4="==========";                                 //Additional Settings
int MagicNumber=-1;                                           //Magic Number For The Orders Opened By This EA
const int MagicNumber_init=MagicNumber;                              // Store the initial value  of MagicNumber
input string HotNotes="";                                           //Additional Comment User can add to the order
string OrderNote="";                                           //Comment For The Orders Opened By This EA
input int Slippage=5;                                                //Slippage in points
input int MaxSpread=100;                                             //Maximum Allowed Spread To Trade In Points

input string Comment_5="==========";                                 //News filter Settings 
input  int AfterNewsStop=5; // Indent after News, minuts
input  int BeforeNewsStop=5; // Indent before News, minuts
input bool NewsLight= true; // Enable light news
input bool NewsMedium=true; // Enable medium news
input bool NewsHard=true; // Enable hard news
input int  offset=2;     // Your Time Zone, GMT (for news)
input string NewsSymb=" USD,EUR,JPY,GBP "; //Currency to display the news (empty - only the current currencies)
input string desc="Core CPI, Nonfarm Payrolls, interest rate, unemployment rate";
input bool  DrawLines=false;       // Draw lines on the chart
input bool  Next           = false;      // Draw only the future of news line
input bool  Signal         = false;      // Signals on the upcoming news

color highc          = clrRed;     // Colour important news
color mediumc        = clrBlue;    // Colour medium news
color lowc           = clrLime;    // The color of weak news
int   Style          = 2;          // Line style
int   Upd            = 86400;      // Period news updates in seconds

bool  Vhigh          = false;
bool  Vmedium        = false;
bool  Vlow           = false;
int   MinBefore=0;
int   MinAfter=0;
string newsTypes[];
int NomNews=0;
string NewsArr[4][1000];
int Now=0;
datetime LastUpd;
string str1;



//-GLOBAL VARIABLES-//
//The variables included in this section are global, hence they can be used in any part of the code
//It is useful to add a comment to remember what is the variable for

bool IsPreChecksOk=false;                 //Indicates if the pre checks are satisfied
bool IsNewCandle=false;                   //Indicates if this is a new candle formed
bool IsSpreadOK=false;                    //Indicates if the spread is low enough to trade
bool IsOperatingHours=false;              //Indicates if it is possible to trade at the current time (server time)
bool IsTradedThisBar=false;               //Indicates if an order was already executed in the current candle

double TickValue=0;                       //Value of a tick in account currency at 1 lot
double LotSize=0;                         //Lot size for the position
int rsi_thresh_exit = 75;
int rsi_thresh_entry = 25;
int OrderOpRetry=10;                      //Number of attempts to retry the order submission
int TotalOpenOrders=0;                    //Number of total open orders
int TotalOpenBuy=0;                       //Number of total open buy orders
int TotalOpenSell=0;                      //Number of total open sell orders
int StopLossBy=SL_BY_POINTS;              //How the stop loss is passed for the lot size calculation

ENUM_SIGNAL_ENTRY SignalEntry=SIGNAL_ENTRY_NEUTRAL;      //Entry signal variable
ENUM_SIGNAL_EXIT SignalExit=SIGNAL_EXIT_NEUTRAL;         //Exit signal variable

bool is_error(string str)
{
    return (StringFind(str, "error") != -1);
}

bool prepare_db()
{
    string test_query = "SELECT * FROM \"Pre_trade\" LIMIT 1";
    string res = pmql_exec(test_query);
    
    if (!is_error(res))
    {
        Print("Table " + "Pre_trade" + " already exists.");
        return (true);
    }
    
    if (StringFind(res, "does not exist") > 0)
    {
        Print("Table " + "Pre_trade" + " does not exist, will create.");
        
        string create_query = "CREATE TABLE \"Pre_trade\" (hour int NOT NULL, volume int NOT NULL, atr numeric(10,5) NOT NULL , \"timestamp\" timestamp with time zone NOT NULL, CONSTRAINT \"Pre_trade_pkey\" PRIMARY KEY (\"timestamp\", hour, volume, atr)) WITH (OIDS=FALSE);";
        res = pmql_exec(create_query);
        
        if (is_error(res))
        {
            Print(res);
            return (false);
        }
        create_query = "CREATE TRIGGER new_row_trigger AFTER INSERT ON \"Pre_trade\" FOR EACH ROW EXECUTE FUNCTION notify_new_row()";
        res = pmql_exec(create_query);
        
        if (is_error(res))
        {
            Print(res);
            return (false);
        }
        return (true);
    }
    else
    {
        Print(res);
        return (false);
    }
    
    return (false);
}

// Function to trim leading and trailing spaces from a string
string Trim(string str) {
   int start = 0;
   int end = StringLen(str) - 1;

   // Trim leading spaces
   while (start <= end && StringGetChar(str, start) == ' ') {
      start++;
   }

   // Trim trailing spaces
   while (end >= start && StringGetChar(str, end) == ' ') {
      end--;
   }

   // Return the trimmed substring
   return StringSubstr(str, start, end - start + 1);
}

// Function to split the input string into an array of news types
void ParseNewsTypes(string desc) {
   int start = 0;
   int pos = StringFind(desc, ",", start);

   while (pos >= 0) {
      string newsType = StringSubstr(desc, start, pos - start);
      ArrayResize(newsTypes, ArraySize(newsTypes) + 1);
      newsTypes[ArraySize(newsTypes) - 1] = Trim(newsType);
      start = pos + 1;
      pos = StringFind(desc, ",", start);
   }

   // Add the last element
   if (start < StringLen(desc)) {
      string newsType = StringSubstr(desc, start);
      ArrayResize(newsTypes, ArraySize(newsTypes) + 1);
      newsTypes[ArraySize(newsTypes) - 1] = Trim(newsType);
   }
}


//-NATIVE MT4 EXPERT ADVISOR RUNNING FUNCTIONS-//

//OnInit is executed once, when the EA is loaded
//OnInit is also executed if the time frame or symbol for the chart is changed
int OnInit(){

   string res = pmql_connect(g_db_ip_setting, g_db_port_setting, g_db_user_setting, g_db_password_setting, g_db_name_setting);
    if ((res != "ok") && (res != "already connected"))
    {
        Print("DB not connected!");
        return (-1);
    }

    if (!prepare_db())
        return (-1);

   //+------------------------------------------------------------------+
  //|   Auto Magic Number feature
  //+------------------------------------------------------------------+
  if(MagicNumber_init < 0 || MagicNumber < 0){ // Prevent Magic Number of -1 bug
    // Unique sting id.    // With respect to unique chart ID and the FirstTradeType
    string id = IntegerToString(ChartID()) + WindowExpertName() + Symbol();// + Period(); // Same magic number for all timeframe
    
    // If there isn't already a Global Variable with the id in wich search for the MagicNumber create it  
    if(!GlobalVariableCheck(id))
    {
    MagicNumber = WindowHandle(Symbol(),0);   //Print((FirstTradeType));
    GlobalVariableSet(id,MagicNumber);
    }
    else // Just get the MagicNumber for the unique id
    {
    MagicNumber = (int)GlobalVariableGet(id);
    }
  }
   //It is useful to set a function to check the integrity of the initial parameters and call it as first thing
   CheckPreChecks();
   //If the initial pre checks have something wrong, stop the program
   if(!IsPreChecksOk){
      OnDeinit(INIT_FAILED);
      return(INIT_FAILED);
   }   
   //Function to initialize the values of the global variables
   InitializeVariables();
   ParseNewsTypes(desc);
   if(rsi_fast_period < 1 || rsi_fast_period > 1000)
   {
      Alert("Invalid RSI fast period");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(start_hour < 0 || start_hour > 24)
   {
      Alert("Invalid Start time");
      return(INIT_PARAMETERS_INCORRECT);
      
   }
   if(end_hour < 0 || end_hour > 24)
   {
      Alert("Invalid End time");
      return(INIT_PARAMETERS_INCORRECT);
      
   }
   if(tp_percent < 0.01 || tp_percent > 100)
    {
        Alert("Invalid take profit percent");
        return(INIT_PARAMETERS_INCORRECT);
    }
    if(sl_percent < 0.01 || sl_percent > 100)
    {
        Alert("Invalid stop loss percent");
        return(INIT_PARAMETERS_INCORRECT);
    }
    if(StringLen(NewsSymb)>1)str1=NewsSymb;
    else str1=Symbol();

    Vhigh=NewsHard;
    Vmedium=NewsMedium;
    Vlow=NewsLight;
    
    MinBefore=BeforeNewsStop;
    MinAfter=AfterNewsStop;
    
     
   
   //If everything is ok the function returns successfully and the control is passed to a timer or the OnTike function
   return(INIT_SUCCEEDED);
}


//The OnDeinit function is called just before terminating the program
void OnDeinit(const int reason){
   //You can include in this function something you want done when the EA closes
   //For example clean the chart form graphical objects, write a report to a file or some kind of alert
   EventKillTimer();
   pmql_disconnect();
   ObjectsDeleteAll(0,OBJ_VLINE);
}


//The OnTick function is triggered every time MT4 receives a price change for the symbol in the chart
void OnTick(){
   Comment("News Check: " + news_filter());
   //Re-initialize the values of the global variables at every run
   InitializeVariables();
   //ScanOrders scans all the open orders and collect statistics, if an error occurs it skips to the next price change
   if(!ScanOrders()) return;
   //CheckNewBar checks if the price change happened at the start of a new bar
   CheckNewBar();
   //CheckOperationHours checks if the current time is in the operating hours
   if(!IsNewCandle) return;
   //CheckOperationHours();
   //CheckSpread checks if the spread is above the maximum spread allowed
   CheckSpread();
   //CheckTradedThisBar checks if there was already a trade executed in the current candle
   CheckTradedThisBar();
   //EvaluateExit contains the code to decide if there is an exit signal
   EvaluateExit();
   //ExecuteExit executes the exit in case there is an exit signal
   ExecuteExit();
   //Scan orders again in case some where closed, if an error occurs it skips to the next price change
   if(!ScanOrders()) return;
   //Execute Trailing Stop
   ExecuteTrailingStop();
   //EvaluateEntry contains the code to decide if there is an entry signal
   EvaluateEntry();
   //ExecuteEntry executes the entry in case there is an entry signal
   ExecuteEntry();
}


//-CUSTOM EA FUNCTIONS-//

//Perform integrity checks when the EA is loaded
void CheckPreChecks(){
   IsPreChecksOk=true;
   //Check if Live Trading is enabled in MT4
   if(!IsTradeAllowed()){
      //IsPreChecksOk=false;
      Print("Live Trading is not enabled, please enable it in MT4 and chart settings");
      return;
   }
   //Check if the default stop loss you are setting in above the minimum and below the maximum
   if(DefaultStopLoss<MinStopLoss || DefaultStopLoss>MaxStopLoss){
      IsPreChecksOk=false;
      Print("Default Stop Loss must be between Minimum and Maximum Stop Loss Allowed");
      return;
   }
   //Check if the default take profit you are setting in above the minimum and below the maximum
   if(DefaultTakeProfit<MinTakeProfit || DefaultTakeProfit>MaxTakeProfit){
      IsPreChecksOk=false;
      Print("Default Take Profit must be between Minimum and Maximum Take Profit Allowed");
      return;
   }
   //Check if the Lot Size is between the minimum and maximum
   if(DefaultLotSize<MinLotSize || DefaultLotSize>MaxLotSize){
      IsPreChecksOk=false;
      Print("Default Lot Size must be between Minimum and Maximum Lot Size Allowed");
      return;
   }
   //Slippage must be >= 0
   if(Slippage<0){
      IsPreChecksOk=false;
      Print("Slippage must be a positive value");
      return;
   }
   //MaxSpread must be >= 0
   if(MaxSpread<0){
      IsPreChecksOk=false;
      Print("Maximum Spread must be a positive value");
      return;
   }
   //MaxRiskPerTrade is a % between 0 and 100
   if(MaxRiskPerTrade<0 || MaxRiskPerTrade>100){
      IsPreChecksOk=false;
      Print("Maximum Risk Per Trade must be a percentage between 0 and 100");
      return;
   }
}


//Initialize variables
void InitializeVariables(){
   IsNewCandle=false;
   IsTradedThisBar=false;
   IsOperatingHours=false;
   IsSpreadOK=false;
   
   LotSize=lot_size;
   // print out the lot size as comment on the chart
   //Comment("Lot Size: ",LotSize);
   TickValue=0;
   
   TotalOpenBuy=0;
   TotalOpenSell=0;
   TotalOpenOrders=0;
   
   SignalEntry=SIGNAL_ENTRY_NEUTRAL;
   SignalExit=SIGNAL_EXIT_NEUTRAL;
}


//Evaluate if there is an entry signal
void EvaluateEntry(){
   SignalEntry=SIGNAL_ENTRY_NEUTRAL;
   //if(!IsSpreadOK) return;    //If the spread is too high don't give an entry signal
   //if(UseTradingHours && !IsOperatingHours) return;      //If you are using trading hours and it's not a trading hour don't give an entry signal
   //if(!IsNewCandle) return;      //If you want to provide a signal only if it's a new candle opening
   //if(IsTradedThisBar) return;   //If you don't want to execute multiple trades in the same bar
   //if(TotalOpenOrders>0) return; //If there are already open orders and you don't want to open more
   
   //This is where you should insert your Entry Signal for BUY orders
   //Include a condition to open a buy order, the condition will have to set SignalEntry=SIGNAL_ENTRY_BUY
   
   //This is where you should insert your Entry Signal for SELL orders
   //Include a condition to open a sell order, the condition will have to set SignalEntry=SIGNAL_ENTRY_SELL
   
   double RSICurr=iRSI(Symbol(),PERIOD_CURRENT,rsi_fast_period, PRICE_OPEN, 1);                 //RSI Current is the RSI value in the last closed candle (1)
   double RSIPrev=iRSI(Symbol(),PERIOD_CURRENT,rsi_fast_period, PRICE_OPEN, 2);                 //RSI Current is the RSI value before the last closed candle (2)
   double atr = iATR(Symbol(),PERIOD_CURRENT, 14, 1);    
   if(RSICurr > rsi_thresh_entry && RSIPrev < rsi_thresh_entry) // open long position
        {
            // open long position
            OrderNote = "(#" + IntegerToString(MagicNumber) + ")" + "TF" + IntegerToString(_Period) + "-" + IntegerToString(OP_BUY) + " " + HotNotes;
            OrderNote = StringSubstr(OrderNote, 0, 31);
            datetime currentTime = TimeCurrent(); // Get the current server time
            int currentHour = TimeHour(currentTime); // Extract the hour from the current time
            
            if(currentHour >= start_hour && currentHour < end_hour && canOpenBasedOnExistingPositions(OP_BUY)){
                // final check for news here
                string news = news_filter();
                if(StringCompare("no_news", news) != 0){
                  Print(news);
                  }
                else{
                   string time = TimeToStr(TimeCurrent(), TIME_DATE | TIME_SECONDS);
                   time = StringSetChar(time, 4, '-');
                   time = StringSetChar(time, 7, '-');
                   
                   int ms = GetTickCount();
                   if (ms < 0) ms = (-1 * ms);
                   ms = ms % 1000;
                   string ms_str = ms;
                   
                   while (StringLen(ms_str) < 3)
                       ms_str = ("0" + ms_str);
               
                   time = time + "." + ms_str;
                   int hour = Hour();
                   int volume = iVolume(Symbol(), PERIOD_CURRENT, 1);
                   string query = "INSERT INTO \"Pre_trade\" (timestamp, hour, volume, atr) VALUES ('" + time + "', " + hour + ", " + volume + ", " + NormalizeDouble(atr, 5) + ")";
                   string res = pmql_exec(query);
                      if (is_error(res))
                          Print(res);
                   Sleep(200);
                   query = "SELECT * FROM \"trade\" WHERE hour = " + hour + " AND volume = " + volume + " AND atr = " + NormalizeDouble(atr, 5) + " ORDER BY timestamp DESC LIMIT 1";
                   res = pmql_exec(query);
                   if (is_error(res))
                       Print(res);
                   
                   Print("Query result = ", res);
                   Print("allow(49)/reject(48) = ", StringGetCharacter(res, StringLen(res)-1));
                   if(StrToInteger(StringGetCharacter(res, StringLen(res)-1)) == 49)
                   {
                     Print("ML filter allowed this trade to be executed");
                     SignalEntry=SIGNAL_ENTRY_BUY;
                   }
                   else{
                     Print("ML filter rejected this trade");
                   
                   }
                }
            }
        }
   else if(RSICurr < 100 - rsi_thresh_entry && RSIPrev > 100 - rsi_thresh_entry) // open short position
        {
            // open short position
            OrderNote = "(#" + IntegerToString(MagicNumber) + ")" + "TF" + IntegerToString(_Period) + "-" + IntegerToString(OP_SELL) + " " + HotNotes;
            OrderNote = StringSubstr(OrderNote, 0, 31);
            datetime currentTime = TimeCurrent(); // Get the current server time
            int currentHour = TimeHour(currentTime); // Extract the hour from the current time
            
            if(currentHour >= start_hour && currentHour < end_hour && canOpenBasedOnExistingPositions(OP_SELL)){
                // final check for news here
                string news = news_filter();
                if(StringCompare("no_news", news) != 0){
                  Print(news);
                 }
                else{
                   string time = TimeToStr(TimeCurrent(), TIME_DATE | TIME_SECONDS);
                   time = StringSetChar(time, 4, '-');
                   time = StringSetChar(time, 7, '-');
                   
                   int ms = GetTickCount();
                   if (ms < 0) ms = (-1 * ms);
                   ms = ms % 1000;
                   string ms_str = ms;
                   
                   while (StringLen(ms_str) < 3)
                       ms_str = ("0" + ms_str);
               
                   time = time + "." + ms_str;
                   int hour = Hour();
                   int volume = iVolume(Symbol(), PERIOD_CURRENT, 1);
                   string query = "INSERT INTO \"Pre_trade\" (timestamp, hour, volume, atr) VALUES ('" + time + "', " + Hour() + ", " + iVolume(Symbol(), PERIOD_CURRENT, 1) + ", " + NormalizeDouble(atr, 5) + ")";
                   string res = pmql_exec(query);
                      if (is_error(res))
                          Print(res);
                   Sleep(200);
                   query = "SELECT * FROM \"trade\" WHERE hour = " + hour + " AND volume = " + volume + " AND atr = " + NormalizeDouble(atr, 5) + " ORDER BY timestamp DESC LIMIT 1";
                   res = pmql_exec(query);
                   if (is_error(res))
                       Print(res);
                
                   Print("Query result = ", res);
                   Print("allow(49)/reject(48) = ", StringGetCharacter(res, StringLen(res)-1)); //48 = 0(reject), 49 = 1(allow)
                   if(StrToInteger(StringGetCharacter(res, StringLen(res)-1)) == 49)
                     {
                      Print("ML filter allowed this trade to be executed");
                      SignalEntry=SIGNAL_ENTRY_SELL;
                     }
                   else{
                     Print("ML filter rejected this trade");
                   
                   }
                   
                   
                }
            }
        }
   
   
}

void PrintType(int value) {
    Print("Type: int");
}

// Function to print type of a double
void PrintType(double value) {
    Print("Type: double");
}

// Function to print type of a string
void PrintType(string value) {
    Print("Type: string");
}

// Function to print type of a boolean
void PrintType(bool value) {
    Print("Type: bool");
}


//Execute entry if there is an entry signal
void ExecuteEntry(){
   //If there is no entry signal no point to continue, exit the function
   if(SignalEntry==SIGNAL_ENTRY_NEUTRAL) return;
   int Operation;
   double OpenPrice=0;
   double StopLossPrice=0;
   double TakeProfitPrice=0;
   //If there is a Buy entry signal
   if(SignalEntry==SIGNAL_ENTRY_BUY){
      RefreshRates();   //Get latest rates
      Operation=OP_BUY; //Set the operation to BUY
      OpenPrice=Ask;    //Set the open price to Ask price
      //If the Stop Loss is fixed and the default stop loss is set
      if(StopLossMode==SL_FIXED && DefaultStopLoss>0){
         StopLossPrice=OpenPrice-DefaultStopLoss*Point;
      }
      //If the Stop Loss is automatic
      if(StopLossMode==SL_AUTO){
         //Set the Stop Loss to the custom stop loss price
         StopLossPrice=StopLossPriceCalculate(OP_BUY);
      }
      //If the Take Profix price is fixed and defined
      if(TakeProfitMode==TP_FIXED && DefaultTakeProfit>0){
         TakeProfitPrice=OpenPrice+DefaultTakeProfit*Point;
      }
      //If the Take Profit is automatic
      if(TakeProfitMode==TP_AUTO){
         //Set the Take Profit to the custom take profit price
         TakeProfitPrice=TakeProfitCalculate(OP_BUY);
      }
      //Normalize the digits for the float numbers
      OpenPrice=NormalizeDouble(OpenPrice,Digits());
      StopLossPrice=NormalizeDouble(StopLossPrice,Digits());
      TakeProfitPrice=NormalizeDouble(TakeProfitPrice,Digits());   
      //Submit the order  
      SendOrder(Operation,Symbol(),OpenPrice,StopLossPrice,TakeProfitPrice);
   }
   if(SignalEntry==SIGNAL_ENTRY_SELL){
      RefreshRates();   //Get latest rates
      Operation=OP_SELL; //Set the operation to SELL
      OpenPrice=Bid;    //Set the open price to Ask price
      //If the Stop Loss is fixed and the default stop loss is set
      if(StopLossMode==SL_FIXED && DefaultStopLoss>0){
         StopLossPrice=OpenPrice+DefaultStopLoss*Point();
      }
      //If the Stop Loss is automatic
      if(StopLossMode==SL_AUTO){
         //Set the Stop Loss to the custom stop loss price
         StopLossPrice=StopLossPriceCalculate(OP_SELL);
      }
      //If the Take Profix price is fixed and defined
      if(TakeProfitMode==TP_FIXED && DefaultTakeProfit>0){
         TakeProfitPrice=OpenPrice-DefaultTakeProfit*Point();
      }
      //If the Take Profit is automatic
      if(TakeProfitMode==TP_AUTO){
         //Set the Take Profit to the custom take profit price
         TakeProfitPrice=TakeProfitCalculate(OP_SELL);
      }
      //Normalize the digits for the float numbers
      OpenPrice=NormalizeDouble(OpenPrice,Digits());
      StopLossPrice=NormalizeDouble(StopLossPrice,Digits());
      TakeProfitPrice=NormalizeDouble(TakeProfitPrice,Digits());   
      //Submit the order  
      SendOrder(Operation,Symbol(),OpenPrice,StopLossPrice,TakeProfitPrice);
   }
   
}


//Evaluate if there is an exit signal
void EvaluateExit(){
   SignalExit=SIGNAL_EXIT_NEUTRAL;
   
   //This is where you should include your exit signal for BUY orders
   //If you want, include a condition to close the open buy orders, condition will have to set SignalExit=SIGNAL_EXIT_BUY then return 

   //This is where you should include your exit signal for SELL orders
   //If you want, include a condition to close the open sell orders, condition will have to set SignalExit=SIGNAL_EXIT_SELL then return 

   //This is where you should include your exit signal for ALL orders
   //If you want, include a condition to close all the open orders, condition will have to set SignalExit=SIGNAL_EXIT_ALL then return 
   double RSICurr=iRSI(Symbol(),PERIOD_CURRENT,rsi_fast_period, PRICE_OPEN,0);                 //RSI Current is the RSI value in the last closed candle (1)
   double RSIPrev=iRSI(Symbol(),PERIOD_CURRENT,rsi_fast_period, PRICE_OPEN,1);
   if(RSICurr > rsi_thresh_exit && RSIPrev < rsi_thresh_exit) // close long position
        {
            // close long positions

            SignalExit=SIGNAL_EXIT_BUY;

            
        }
        else if(RSICurr < 100 - rsi_thresh_exit && RSIPrev > 100 - rsi_thresh_exit) // close short position
        {
            // close all short positions
            SignalExit=SIGNAL_EXIT_SELL;     
        }
}


//Execute exit if there is an exit signal
void ExecuteExit(){
   //If there is no Exit Signal no point to continue the routine
   if(SignalExit==SIGNAL_EXIT_NEUTRAL) return;
   //If there is an exit signal for all orders
   if(SignalExit==SIGNAL_EXIT_ALL){
      //Close all orders
      CloseAll(OP_ALL);
   }
   //If there is an exit signal for BUY order
   if(SignalExit==SIGNAL_EXIT_BUY){
      //Close all BUY orders
      CloseAll(OP_BUY);
   }
   //If there is an exit signal for SELL orders
   if(SignalExit==SIGNAL_EXIT_SELL){
      //Close all SELL orders
      CloseAll(OP_SELL);
   }

}


//Execute Trailing Stop to limit losses and lock in profits
void ExecuteTrailingStop(){
   //If the option is off then exit
   if(!UseTrailingStop) return;
   //If there are no open orders no point to continue the code
   if(TotalOpenOrders==0) return;
   //if(!IsNewCandle) return;      //If you only want to do the stop trailing once at the beginning of a new candle
   //Scan all the orders to see if some needs a stop loss update
   for(int i=0;i<OrdersTotal();i++) {
      //If there is a problem reading the order print the error, exit the function and return false
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false){
         int Error=GetLastError();
         string ErrorText=GetLastErrorText(Error);
         Print("ERROR - Unable to select the order - ",Error," - ",ErrorText);
         return;
      }
      //If the order is not for the instrument on chart we can ignore it
      if(OrderSymbol()!=Symbol()) continue;
      //If the order has Magic Number different from the Magic Number of the EA then we can ignore it
      if(OrderMagicNumber()!=MagicNumber) continue;
      //Define current values
      RefreshRates();
      double SLPrice=NormalizeDouble(OrderStopLoss(),Digits());     //Current Stop Loss price for the order
      double TPPrice=NormalizeDouble(OrderTakeProfit(),Digits());   //Current Take Profit price for the order
      double Spread=MarketInfo(Symbol(),MODE_SPREAD)*Point();       //Current Spread for the instrument
      double StopLevel=MarketInfo(Symbol(),MODE_STOPLEVEL)*Point(); //Minimum distance between current price and stop loss

      //If it is a buy order then trail stop for buy orders
      if(OrderType()==OP_BUY){
         //Include code to trail the stop for buy orders
         double NewSLPrice=0;
         
         //This is where you should include the code to assign a new value to the STOP LOSS
         
         
         double NewTPPrice=TPPrice;
         //Normalize the price before the submission
         NewSLPrice=NormalizeDouble(NewSLPrice,Digits());
         //If there is no new stop loss set then skip to next order
         if(NewSLPrice==0) continue;
         //If the new stop loss price is lower than the previous then skip to next order, we only move the stop closer to the price and not further away
         if(NewSLPrice<=SLPrice) continue;
         //If the distance between the current price and the new stop loss is not enough then skip to next order
         //This allows to avoid error 130 when trying to update the order
         if(Bid-NewSLPrice<StopLevel) continue;
         //Submit the update
         ModifyOrder(OrderTicket(),OrderOpenPrice(),NewSLPrice,NewTPPrice);         
      }
      //If it is a sell order then trail stop for sell orders
      if(OrderType()==OP_SELL){
         //Include code to trail the stop for sell orders
         double NewSLPrice=0;
         
         //This is where you should include the code to assign a new value to the STOP LOSS
         
         
         double NewTPPrice=TPPrice;
         //Normalize the price before the submission
         NewSLPrice=NormalizeDouble(NewSLPrice,Digits());
         //If there is no new stop loss set then skip to next order
         if(NewSLPrice==0) continue;
         //If the new stop loss price is higher than the previous then skip to next order, we only move the stop closer to the price and not further away
         if(NewSLPrice>=SLPrice) continue;
         //If the distance between the current price and the new stop loss is not enough then skip to next order
         //This allows to avoid error 130 when trying to update the order
         if(NewSLPrice-Ask<StopLevel) continue;
         //Submit the update
         ModifyOrder(OrderTicket(),OrderOpenPrice(),NewSLPrice,NewTPPrice);         
      }
   }
   return;
}


//Check and return if the spread is not too high
void CheckSpread(){
   //Get the current spread in points, the (int) transforms the double coming from MarketInfo into an integer to avoid a warning when compiling
   int SpreadCurr=(int)MarketInfo(Symbol(),MODE_SPREAD);
   if(SpreadCurr<=MaxSpread){
      IsSpreadOK=true;
   }
   else{
      IsSpreadOK=false;
   }
}


//Check and return if it is operation hours or not
void CheckOperationHours(){
   //If we are not using operating hours then IsOperatingHours is true and I skip the other checks
   if(!UseTradingHours){
      IsOperatingHours=true;
      return;
   }
   //Check if the current hour is between the allowed hours of operations, if so IsOperatingHours is set true
   if(TradingHourStart==TradingHourEnd && Hour()==TradingHourStart) IsOperatingHours=true;
   if(TradingHourStart<TradingHourEnd && Hour()>=TradingHourStart && Hour()<=TradingHourEnd) IsOperatingHours=true;
   if(TradingHourStart>TradingHourEnd && ((Hour()>=TradingHourStart && Hour()<=23) || (Hour()<=TradingHourEnd && Hour()>=0))) IsOperatingHours=true;
}


//Check if it is a new bar
datetime NewBarTime=TimeCurrent();
void CheckNewBar(){
   //NewBarTime contains the open time of the last bar known
   //if that open time is the same as the current bar then we are still in the current bar, otherwise we are in a new bar
   if(NewBarTime==iTime(Symbol(),PERIOD_CURRENT,0)) IsNewCandle=false;
   else{
      NewBarTime=iTime(Symbol(),PERIOD_CURRENT,0);
      IsNewCandle=true;
   }
}


//Check if there was already an order open this bar
datetime LastBarTraded;
void CheckTradedThisBar(){
   //LastBarTraded contains the open time the last trade
   //if that open time is in the same bar as the current then IsTradedThisBar is true
   if(iBarShift(Symbol(),PERIOD_CURRENT,LastBarTraded)==0) IsTradedThisBar=true;
   else IsTradedThisBar=false;
}


//Lot Size Calculator
void LotSizeCalculate(double SL=0){
   //If the position size is dynamic
   if(RiskDefaultSize==RISK_DEFAULT_AUTO){
      //If the stop loss is not zero then calculate the lot size
      if(SL!=0){
         double RiskBaseAmount=0;
         //TickValue is the value of the individual price increment for 1 lot of the instrument, expressed in the account currenty
         TickValue=MarketInfo(Symbol(),MODE_TICKVALUE);    
         //Define the base for the risk calculation depending on the parameter chosen    
         if(RiskBase==RISK_BASE_BALANCE) RiskBaseAmount=AccountBalance();
         if(RiskBase==RISK_BASE_EQUITY) RiskBaseAmount=AccountEquity();
         if(RiskBase==RISK_BASE_FREEMARGIN) RiskBaseAmount=AccountFreeMargin();
         //Calculate the Position Size
         LotSize=(RiskBaseAmount*MaxRiskPerTrade/100)/(SL*TickValue);
      }
      //If the stop loss is zero then the lot size is the default one
      if(SL==0){
         LotSize=DefaultLotSize;
      }
   }
   //Normalize the Lot Size to satisfy the allowed lot increment and minimum and maximum position size
   LotSize=MathFloor(LotSize/MarketInfo(Symbol(),MODE_LOTSTEP))*MarketInfo(Symbol(),MODE_LOTSTEP);
   //Limit the lot size in case it is greater than the maximum allowed by the user
   if(LotSize>MaxLotSize) LotSize=MaxLotSize;
   //Limit the lot size in case it is greater than the maximum allowed by the broker
   if(LotSize>MarketInfo(Symbol(),MODE_MAXLOT)) LotSize=MarketInfo(Symbol(),MODE_MAXLOT);
   //If the lot size is too small then set it to 0 and don't trade
   if(LotSize<MinLotSize || LotSize<MarketInfo(Symbol(), MODE_MINLOT)) LotSize=0;
}


//Stop Loss Price Calculation if dynamic
double StopLossPriceCalculate(int Command=-1){
   double StopLossPrice=0;
   //Include a value for the stop loss, ideally coming from an indicator
   if(Command==OP_BUY){
      //Include code to calculate the stop loss for buy orders
      StopLossPrice = Ask - (Ask * sl_percent / 100);

   }
   else if(Command==OP_SELL){
      //Include code to calculate the stop loss for sell orders
      StopLossPrice = Bid + (Bid * sl_percent / 100);
   }
   else{
      //Include code to calculate the stop loss for pending orders
      Print("Stop Loss Price Calculation for Pending Orders not implemented yet");
   }
   return StopLossPrice;
}


//Take Profit Price Calculation if dynamic
double TakeProfitCalculate(int Command=-1){
   double TakeProfitPrice=0;
   //Include a value for the take profit, ideally coming from an indicator
    if(Command==OP_BUY){
        //Include code to calculate the take profit for buy orders
        TakeProfitPrice = Ask + (Ask * tp_percent / 100);
    }
    else if(Command==OP_SELL){
        //Include code to calculate the take profit for sell orders
        TakeProfitPrice = Bid - (Bid * tp_percent / 100);
    }
    else{
        //Include code to calculate the take profit for pending orders
        Print("Take Profit Price Calculation for Pending Orders not implemented yet");
    }
   return TakeProfitPrice;
}


//Send Order Function adjusted to handle errors and retry multiple times
void SendOrder(int Command, string Instrument, double OpenPrice, double SLPrice, double TPPrice, datetime Expiration=0){
   //Retry a number of times in case the submission fails
   for(int i=1; i<=OrderOpRetry; i++){
      //Set the color for the open arrow for the order
      color OpenColor=clrBlueViolet;
      if(Command==OP_BUY){
         OpenColor=clrChartreuse;
      }
      if(Command==OP_SELL){
         OpenColor=clrDarkTurquoise;
      }
      //Calculate the position size, if the lot size is zero then exit the function
      double SLPoints=0;
      //If the Stop Loss price is set then find the points of distance between open price and stop loss price, and round it
      if(SLPrice>0) SLPoints=MathCeil(MathAbs(OpenPrice-SLPrice)/Point());
      //Call the function to calculate the position size
      LotSizeCalculate(SLPoints);
      //If the position size is zero then exit and don't submit any orderInit
      if(LotSize==0) return;
      //Submit the order
      int res=OrderSend(Instrument,Command,LotSize,OpenPrice,Slippage,NormalizeDouble(SLPrice,Digits()),NormalizeDouble(TPPrice,Digits()),OrderNote,MagicNumber,Expiration,OpenColor);
      //If the submission is successful print it in the log and exit the function
      if(res!=-1){
         Print("TRADE - OPEN SUCCESS - Order ",res," submitted: Command ",Command," Volume ",LotSize," Open ",OpenPrice," Stop ",SLPrice," Take ",TPPrice," Expiration ",Expiration);
         break;
      }
      //If the submission failed print the error
      else{
         Print("TRADE - OPEN FAILED - Order ",res," submitted: Command ",Command," Volume ",LotSize," Open ",OpenPrice," Stop ",SLPrice," Take ",TPPrice," Expiration ",Expiration);
         int Error=GetLastError();
         string ErrorText=GetLastErrorText(Error);
         Print("ERROR - NEW - error sending order, return error: ",Error," - ",ErrorText);
      } 
   }
   return;
}


//Modify Order Function adjusted to handle errors and retry multiple times
void ModifyOrder(int Ticket, double OpenPrice, double SLPrice, double TPPrice){
   //Try to select the order by ticket number and print the error if failed
   if(OrderSelect(Ticket,SELECT_BY_TICKET)==false){
      int Error=GetLastError();
      string ErrorText=GetLastErrorText(Error);
      Print("ERROR - SELECT TICKET - error selecting order ",Ticket," return error: ",Error);
      return;
   }
   //Normalize the digits for stop loss and take profit price
   SLPrice=NormalizeDouble(SLPrice,Digits());
   TPPrice=NormalizeDouble(TPPrice,Digits());
   //Try to submit the changes multiple times
   for(int i=1; i<=OrderOpRetry; i++){
      //Submit the change
      bool res=OrderModify(Ticket,OpenPrice,SLPrice,TPPrice,0,Blue);
      //If the change is successful print the result and exit the function
      if(res){
         Print("TRADE - UPDATE SUCCESS - Order ",Ticket," new stop loss ",SLPrice," new take profit ",TPPrice);
         break;
      }
      //If the change failed print the error with additional information to troubleshoot
      else{
         int Error=GetLastError();
         string ErrorText=GetLastErrorText(Error);
         Print("ERROR - UPDATE FAILED - error modifying order ",Ticket," return error: ",Error," Open=",OpenPrice,
               " Old SL=",OrderStopLoss()," Old TP=",OrderTakeProfit(),
               " New SL=",SLPrice," New TP=",TPPrice," Bid=",MarketInfo(OrderSymbol(),MODE_BID)," Ask=",MarketInfo(OrderSymbol(),MODE_ASK));
         Print("ERROR - ",ErrorText);
      } 
   }
   return;
}


//Close Single Order Function adjusted to handle errors and retry multiple times
void CloseOrder(int Ticket, double Lots, double CurrentPrice){
   //Try to close the order by ticket number multiple times in case of failure
   for(int i=1; i<=OrderOpRetry; i++){
      //Send the close command
      bool res=OrderClose(Ticket,Lots,CurrentPrice,Slippage,Red);
      //If the close was successful print the resul and exit the function
      if(res){
         Print("TRADE - CLOSE SUCCESS - Order ",Ticket," closed at price ",CurrentPrice);
         break;
      }
      //If the close failed print the error
      else{
         int Error=GetLastError();
         string ErrorText=GetLastErrorText(Error);
         Print("ERROR - CLOSE FAILED - error closing order ",Ticket," return error: ",Error," - ",ErrorText);
      } 
   }
   return;
}


//Close All Orders of a specified type
const int OP_ALL=-1; //Constant to define the additional OP_ALL command which is the reference to all type of orders
void CloseAll(int Command){
   //If the command is OP_ALL then run the CloseAll function for both BUY and SELL orders
   if(Command==OP_ALL){
      CloseAll(OP_BUY);
      CloseAll(OP_SELL);
      return;
   }
   double ClosePrice=0;
   //Scan all the orders to close them individually
   //NOTE that the for loop scans from the last to the first, this is because when we close orders the list of orders is updated
   //hence the for loop would skip orders if we scan from first to last
   for(int i=OrdersTotal()-1; i>=0; i--) {
      //First select the order individually to get its details, if the selection fails print the error and exit the function
      if( OrderSelect( i, SELECT_BY_POS, MODE_TRADES ) == false ) {
         Print("ERROR - Unable to select the order - ",GetLastError());
         break;
      }
      //Check if the order is for the current symbol and was opened by the EA and is the type to be closed
      if(OrderMagicNumber()==MagicNumber && OrderSymbol()==Symbol() && OrderType()==Command) {
         //Define the close price
         RefreshRates();
         if(Command==OP_BUY) ClosePrice=Bid;
         if(Command==OP_SELL) ClosePrice=Ask;
         //Get the position size and the order identifier (ticket)
         double Lots=OrderLots();
         int Ticket=OrderTicket();
         //Close the individual order
         CloseOrder(Ticket,Lots,ClosePrice);
      }
   }
}


//Scan all orders to find the ones submitted by the EA
//NOTE This function is defined as bool because we want to return true if it is successful and false if it fails
bool ScanOrders(){
   //Scan all the orders, retrieving some of the details
   for(int i=0;i<OrdersTotal();i++) {
      //If there is a problem reading the order print the error, exit the function and return false
      if(OrderSelect(i,SELECT_BY_POS,MODE_TRADES)==false){
         int Error=GetLastError();
         string ErrorText=GetLastErrorText(Error);
         Print("ERROR - Unable to select the order - ",Error," - ",ErrorText);
         return false;
      }
      //If the order is not for the instrument on chart we can ignore it
      if(OrderSymbol()!=Symbol()) continue;
      //If the order has Magic Number different from the Magic Number of the EA then we can ignore it
      if(OrderMagicNumber()!=MagicNumber) continue;
      //If it is a buy order then increment the total count of buy orders
      if(OrderType()==OP_BUY) TotalOpenBuy++;
      //If it is a sell order then increment the total count of sell orders
      if(OrderType()==OP_SELL) TotalOpenSell++;
      //Increment the total orders count
      TotalOpenOrders++;
      //Find what is the open time of the most recent trade and assign it to LastBarTraded
      //this is necessary to check if we already traded in the current candle
      if(OrderOpenTime()>LastBarTraded || LastBarTraded==0) LastBarTraded=OrderOpenTime();
   }
   return true;
}


bool canOpenBasedOnExistingPositions(int direction){
  int totalPositions = OrdersTotal();
  int matchingPositions = 0;
  double last_traded_price = 0;
  int last_traded_direction = -5; //OP_BUY = 0, OP_SELL = 1
  for (int i = 0; i < totalPositions; i++)
  {
    if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)){
      Alert("Error selecting order ", i, " ", GetLastErrorText(GetLastError()));
      return false;
    }
    if(OrderMagicNumber() != MagicNumber){
      continue;
    }
    // get the last traded price and direction
    if(OrderOpenTime() > last_traded_price){
      last_traded_price = OrderOpenPrice();
      last_traded_direction = OrderType();
    }
    matchingPositions++;
  }

  

  if(matchingPositions > 0){ // if there are positions
    if(last_traded_direction == direction){ //check if the last traded direction is the same as the current direction
      if(direction == OP_BUY){
        return (Ask < last_traded_price && matchingPositions < 1); // if the direction is buy, then the current ask price should be lower than the last traded price
      }
      else{ // OP_SELL
        return (Bid > last_traded_price && matchingPositions < 1); // if the direction is sell, then the current Bid price should be higher than the last traded price
      }
    }
    else if(last_traded_direction == -5){ // if the last traded direction is not set (first trade)
      Alert("Last traded direction is not set. This should not happen. Please contact the developer.");
      return matchingPositions < 1;
    }
    else{ // conflicting direction (opposite direction from existing positions)
      // close all positions
      CloseAll(OP_ALL);
      // open new position
      return matchingPositions < 1;
    }
  }
  else{
    return matchingPositions < 1; // no positions
  }
}

string GetErrorDescription(int error)
  {
// Type: Fixed Template 
// Do not edit unless you know what you're doing

// This function returns the exact error

   string ErrorDescription="";
//---
   switch(error)
     {
      case 0:     ErrorDescription = "NO Error. Everything should be good.";                                    break;
      case 1:     ErrorDescription = "No error returned, but the result is unknown";                            break;
      case 2:     ErrorDescription = "Common error";                                                            break;
      case 3:     ErrorDescription = "Invalid trade parameters";                                                break;
      case 4:     ErrorDescription = "Trade server is busy";                                                    break;
      case 5:     ErrorDescription = "Old version of the client terminal";                                      break;
      case 6:     ErrorDescription = "No connection with trade server";                                         break;
      case 7:     ErrorDescription = "Not enough rights";                                                       break;
      case 8:     ErrorDescription = "Too frequent requests";                                                   break;
      case 9:     ErrorDescription = "Malfunctional trade operation";                                           break;
      case 64:    ErrorDescription = "Account disabled";                                                        break;
      case 65:    ErrorDescription = "Invalid account";                                                         break;
      case 128:   ErrorDescription = "Trade timeout";                                                           break;
      case 129:   ErrorDescription = "Invalid price";                                                           break;
      case 130:   ErrorDescription = "Invalid stops";                                                           break;
      case 131:   ErrorDescription = "Invalid trade volume";                                                    break;
      case 132:   ErrorDescription = "Market is closed";                                                        break;
      case 133:   ErrorDescription = "Trade is disabled";                                                       break;
      case 134:   ErrorDescription = "Not enough money";                                                        break;
      case 135:   ErrorDescription = "Price changed";                                                           break;
      case 136:   ErrorDescription = "Off quotes";                                                              break;
      case 137:   ErrorDescription = "Broker is busy";                                                          break;
      case 138:   ErrorDescription = "Requote";                                                                 break;
      case 139:   ErrorDescription = "Order is locked";                                                         break;
      case 140:   ErrorDescription = "Long positions only allowed";                                             break;
      case 141:   ErrorDescription = "Too many requests";                                                       break;
      case 145:   ErrorDescription = "Modification denied because order too close to market";                   break;
      case 146:   ErrorDescription = "Trade context is busy";                                                   break;
      case 147:   ErrorDescription = "Expirations are denied by broker";                                        break;
      case 148:   ErrorDescription = "Too many open and pending orders (more than allowed)";                    break;
      case 4000:  ErrorDescription = "No error";                                                                break;
      case 4001:  ErrorDescription = "Wrong function pointer";                                                  break;
      case 4002:  ErrorDescription = "Array index is out of range";                                             break;
      case 4003:  ErrorDescription = "No memory for function call stack";                                       break;
      case 4004:  ErrorDescription = "Recursive stack overflow";                                                break;
      case 4005:  ErrorDescription = "Not enough stack for parameter";                                          break;
      case 4006:  ErrorDescription = "No memory for parameter string";                                          break;
      case 4007:  ErrorDescription = "No memory for temp string";                                               break;
      case 4008:  ErrorDescription = "Not initialized string";                                                  break;
      case 4009:  ErrorDescription = "Not initialized string in array";                                         break;
      case 4010:  ErrorDescription = "No memory for array string";                                              break;
      case 4011:  ErrorDescription = "Too long string";                                                         break;
      case 4012:  ErrorDescription = "Remainder from zero divide";                                              break;
      case 4013:  ErrorDescription = "Zero divide";                                                             break;
      case 4014:  ErrorDescription = "Unknown command";                                                         break;
      case 4015:  ErrorDescription = "Wrong jump (never generated error)";                                      break;
      case 4016:  ErrorDescription = "Not initialized array";                                                   break;
      case 4017:  ErrorDescription = "DLL calls are not allowed";                                               break;
      case 4018:  ErrorDescription = "Cannot load library";                                                     break;
      case 4019:  ErrorDescription = "Cannot call function";                                                    break;
      case 4020:  ErrorDescription = "Expert function calls are not allowed";                                   break;
      case 4021:  ErrorDescription = "Not enough memory for temp string returned from function";                break;
      case 4022:  ErrorDescription = "System is busy (never generated error)";                                  break;
      case 4050:  ErrorDescription = "Invalid function parameters count";                                       break;
      case 4051:  ErrorDescription = "Invalid function parameter value";                                        break;
      case 4052:  ErrorDescription = "String function internal error";                                          break;
      case 4053:  ErrorDescription = "Some array error";                                                        break;
      case 4054:  ErrorDescription = "Incorrect series array using";                                            break;
      case 4055:  ErrorDescription = "Custom indicator error";                                                  break;
      case 4056:  ErrorDescription = "Arrays are incompatible";                                                 break;
      case 4057:  ErrorDescription = "Global variables processing error";                                       break;
      case 4058:  ErrorDescription = "Global variable not found";                                               break;
      case 4059:  ErrorDescription = "Function is not allowed in testing mode";                                 break;
      case 4060:  ErrorDescription = "Function is not confirmed";                                               break;
      case 4061:  ErrorDescription = "Send mail error";                                                         break;
      case 4062:  ErrorDescription = "String parameter expected";                                               break;
      case 4063:  ErrorDescription = "Integer parameter expected";                                              break;
      case 4064:  ErrorDescription = "Double parameter expected";                                               break;
      case 4065:  ErrorDescription = "Array as parameter expected";                                             break;
      case 4066:  ErrorDescription = "Requested history data in updating state";                                break;
      case 4067:  ErrorDescription = "Some error in trading function";                                          break;
      case 4099:  ErrorDescription = "End of file";                                                             break;
      case 4100:  ErrorDescription = "Some file error";                                                         break;
      case 4101:  ErrorDescription = "Wrong file name";                                                         break;
      case 4102:  ErrorDescription = "Too many opened files";                                                   break;
      case 4103:  ErrorDescription = "Cannot open file";                                                        break;
      case 4104:  ErrorDescription = "Incompatible access to a file";                                           break;
      case 4105:  ErrorDescription = "No order selected";                                                       break;
      case 4106:  ErrorDescription = "Unknown symbol";                                                          break;
      case 4107:  ErrorDescription = "Invalid price";                                                           break;
      case 4108:  ErrorDescription = "Invalid ticket";                                                          break;
      case 4109:  ErrorDescription = "EA is not allowed to trade is not allowed. ";                             break;
      case 4110:  ErrorDescription = "Longs are not allowed. Check the expert properties";                      break;
      case 4111:  ErrorDescription = "Shorts are not allowed. Check the expert properties";                     break;
      case 4200:  ErrorDescription = "Object exists already";                                                   break;
      case 4201:  ErrorDescription = "Unknown object property";                                                 break;
      case 4202:  ErrorDescription = "Object does not exist";                                                   break;
      case 4203:  ErrorDescription = "Unknown object type";                                                     break;
      case 4204:  ErrorDescription = "No object name";                                                          break;
      case 4205:  ErrorDescription = "Object coordinates error";                                                break;
      case 4206:  ErrorDescription = "No specified subwindow";                                                  break;
      case 4207:  ErrorDescription = "Some error in object function";                                           break;
      default:    ErrorDescription = "No error or error is unknown";
     }
   return(ErrorDescription);
  }

string ReadCBOE()
  {

   string cookie=NULL,headers;
   string reqheaders="User-Agent: Mozilla/4.0\r\n";
   char post[],result[];     string TXT="";
   int res;
//--- to work with the server, you must add the URL "https://www.google.com/finance"  
//--- the list of allowed URL (Main menu-> Tools-> Settings tab "Advisors"):
   string google_url="https://ec.forexprostools.com/?columns=exc_currency,exc_importance&amp;importance=1,2,3&calType=week&timeZone=15&lang=1";
//---
   ResetLastError();
//--- download html-pages
   int timeout=5000; //--- timeout less than 1,000 (1 sec.) is insufficient at a low speed of the Internet
   //res=WebRequest("GET",google_url,cookie,NULL,timeout,post,0,result,headers);
   res = WebRequest("GET",google_url,reqheaders,timeout,post,result,headers);
//--- error checking
   if(res==-1)
     {
      Print("WebRequest error, err.code  =",GetLastError());
      MessageBox("You must add the address ' "+google_url+"' in the list of allowed URL tab 'Advisors' "," Error ",MB_ICONINFORMATION);
      //--- You must add the address ' "+ google url"' in the list of allowed URL tab 'Advisors' "," Error "
     }
   else
     {
      //--- successful download
      //PrintFormat("File successfully downloaded, the file size in bytes  =%d.",ArraySize(result));
      //--- save the data in the file
      int filehandle=FileOpen("news-log.html",FILE_WRITE|FILE_BIN);
      //--- проверка ошибки
      if(filehandle!=INVALID_HANDLE)
        {
         //---save the contents of the array result [] in file
         FileWriteArray(filehandle,result,0,ArraySize(result));
         //--- close file
         FileClose(filehandle);

         int filehandle2=FileOpen("news-log.html",FILE_READ|FILE_BIN);
         TXT=FileReadString(filehandle2,ArraySize(result));
         FileClose(filehandle2);
        }else{
         Print("Error in FileOpen. Error code =",GetLastError());
        }
     }

   return(TXT);
  }
//+------------------------------------------------------------------+
datetime TimeNewsFunck(int nomf)
  {
   string s=NewsArr[0][nomf];
   string time=StringConcatenate(StringSubstr(s,0,4),".",StringSubstr(s,5,2),".",StringSubstr(s,8,2)," ",StringSubstr(s,11,2),":",StringSubstr(s,14,4));
   return((datetime)(StringToTime(time) + offset*3600));
  }
//////////////////////////////////////////////////////////////////////////////////
void UpdateNews()
  {
   string TEXT=ReadCBOE();
   
   int sh = StringFind(TEXT,"pageStartAt>")+12;
   int sh2= StringFind(TEXT,"</tbody>");
   TEXT=StringSubstr(TEXT,sh,sh2-sh);
   //Print(TEXT);

   sh=0;
   while(!IsStopped())
     {
      sh = StringFind(TEXT,"event_timestamp",sh)+17;
      sh2= StringFind(TEXT,"onclick",sh)-2;
      if(sh<17 || sh2<0)break;
      NewsArr[0][NomNews]=StringSubstr(TEXT,sh,sh2-sh);

      sh = StringFind(TEXT,"flagCur",sh)+10;
      sh2= sh+3;
      if(sh<10 || sh2<3)break;
      NewsArr[1][NomNews]=StringSubstr(TEXT,sh,sh2-sh);
      if(StringFind(str1,NewsArr[1][NomNews])<0)continue;

      sh = StringFind(TEXT,"title",sh)+7;
      sh2= StringFind(TEXT,"Volatility",sh)-1;
      if(sh<7 || sh2<0)break;
      NewsArr[2][NomNews]=StringSubstr(TEXT,sh,sh2-sh);
      if(StringFind(NewsArr[2][NomNews],"High")>=0 && !Vhigh)continue;
      if(StringFind(NewsArr[2][NomNews],"Moderate")>=0 && !Vmedium)continue;
      if(StringFind(NewsArr[2][NomNews],"Low")>=0 && !Vlow)continue;

      sh=StringFind(TEXT,"left event",sh)+12;
      int sh1=StringFind(TEXT,"Speaks",sh);
      sh2=StringFind(TEXT,"<",sh);
      if(sh<12 || sh2<0)break;
      if(sh1<0 || sh1>sh2)NewsArr[3][NomNews]=StringSubstr(TEXT,sh,sh2-sh);
      else NewsArr[3][NomNews]=StringSubstr(TEXT,sh,sh1-sh);

      NomNews++;
      if(NomNews==300)break;
     }
  }
  
string news_filter()
{
   double CheckNews=0;
   string got_news = "no_news";
   if(AfterNewsStop>0)
     {
      if(TimeCurrent()-LastUpd>=Upd){Comment("News Loading...");Print("News Loading...");UpdateNews();LastUpd=TimeCurrent();Comment("");}
      WindowRedraw();
      //---Draw a line on the chart news--------------------------------------------
      if(DrawLines)
        {
        //Print("length of NomNews is " + NomNews);
         for(int i=0;i<NomNews;i++)
           {
            string Name=StringSubstr(TimeToStr(TimeNewsFunck(i),TIME_MINUTES)+"_"+NewsArr[1][i]+"_"+NewsArr[3][i],0,63);
            //Print("NewsArr[1][i] " + NewsArr[1][i] );
            if(NewsArr[3][i]!="")if(ObjectFind(Name)==0)continue;
            if(StringFind(str1,NewsArr[1][i])<0)continue;
            if(TimeNewsFunck(i)<TimeCurrent() && Next)continue;

            color clrf = clrNONE;
            if(Vhigh && StringFind(NewsArr[2][i],"High")>=0)clrf=highc;
            if(Vmedium && StringFind(NewsArr[2][i],"Moderate")>=0)clrf=mediumc;
            if(Vlow && StringFind(NewsArr[2][i],"Low")>=0)clrf=lowc;

            if(clrf==clrNONE)continue;

            if(NewsArr[3][i]!="")
              {
               ObjectCreate(Name,0,OBJ_VLINE,TimeNewsFunck(i),0);
               ObjectSet(Name,OBJPROP_COLOR,clrf);
               ObjectSet(Name,OBJPROP_STYLE,Style);
               ObjectSetInteger(0,Name,OBJPROP_BACK,true);
              }
           }
        }
      //---------------event Processing------------------------------------
      int i;
      CheckNews=0;
      //Print("length of NomNews is " + NomNews);
      for(i=0;i<NomNews;i++)
        {
         datetime newsTime = TimeNewsFunck(i);
         datetime currentTime = TimeCurrent();
         
         // Extract day, month, and year from news time
         int newsDay = TimeDay(newsTime);
         int newsMonth = TimeMonth(newsTime);
         int newsYear = TimeYear(newsTime);
         
         // Extract day, month, and year from current time
         int currentDay = TimeDay(currentTime);
         int currentMonth = TimeMonth(currentTime);
         int currentYear = TimeYear(currentTime);
         
         // Check if the news date is the same as today's date
         if (newsDay == currentDay && newsMonth == currentMonth && newsYear == currentYear)
         {
            //Print(NewsArr[1][i]);
            if(StringFind(NewsSymb, NewsArr[1][i]) > 0){
               //Print(NewsArr[1][i]);
               //Print("length of Newstype : " + ArraySize(newsTypes));
               for (int j = 0; j < ArraySize(newsTypes); j++) {
                  //Print(newsTypes[j]);
                  if (StringFind(NewsArr[3][i], newsTypes[j]) > 0) {
                     got_news = newsTypes[j] + " detected, trades are filtered.";
                     //Print(got_news);
                     break; // Exit the loop if a match is found
                  }
               }
//               if(StringFind(NewsArr[3][i], "Nonfarm Payrolls")> 0){
//                  got_news = "nonfarm payroll detected, trades is filtered.";
//
//               }
//               else if(StringFind(NewsArr[3][i], "Core CPI")> 0){
//                  got_news = "Core CPI detected, trades is filtered.";
//
//               }
//               else if(StringFind(NewsArr[3][i], "Crude Oil Inventories")> 0){
//                  got_news = "Crude Oil Inventory detected, trades is filtered.";
//               }
               //else if(StringFind(NewsArr[3][i], "PMI")> 0){
               //   got_news = "PMI detected, trades is filtered.";
               //}
            }
         }
         }
         


          
         
            
         
         //int power=0;
         //if(Vhigh && StringFind(NewsArr[2][i],"High")>=0)power=1;
         //if(Vmedium && StringFind(NewsArr[2][i],"Moderate")>=0)power=2;
         //if(Vlow && StringFind(NewsArr[2][i],"Low")>=0)power=3;
         //if(power==0)continue;
         //if(TimeCurrent()+MinBefore*60>TimeNewsFunck(i) && TimeCurrent()-MinAfter*60<TimeNewsFunck(i) && StringFind(str1,NewsArr[1][i])>=0)
           //{
           // CheckNews=1;
            //break;
          // }
        // else CheckNews=0;

        }
      //if(CheckNews==1 && i!=Now && Signal) { Alert("In ",(int)(TimeNewsFunck(i)-TimeCurrent())/60," minutes released news ",NewsArr[1][i],"_",NewsArr[3][i]);Now=i;}
/***  ***/
     //}

//   if(CheckNews>0)
//     {
//      /////  We are doing here if we are in the framework of the news
//      Comment("News time");
//      Print("got news");
//
//     }else{
//      // We are out of scope of the news release (No News)
//      Comment("No news");

//     }
      return got_news;
}