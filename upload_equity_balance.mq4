//+------------------------------------------------------------------+
//|                                        upload_equity_balance.mq4 |
//|                                  Copyright 2024, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"

#include <postgremql4.mqh>


string g_db_ip_setting = "localhost";
string g_db_port_setting = "5432";
string g_db_user_setting = "postgres";
string g_db_password_setting = "1234";
string g_db_name_setting = "postgres";

//Current time zone of your broker (you should clarify this with your broker, +01 is just an example),
//please remember to change it when daylight savings are in effect, of course *only* if your broker observes DST
string g_timezone_setting = "+01";

bool is_error(string str)
{
    return (StringFind(str, "error") != -1);
}

bool prepare_db()
{
    string test_query = "SELECT * FROM \"Equity_balance\" LIMIT 1";
    string res = pmql_exec(test_query);
    
    if (!is_error(res))
    {
        Print("Table " + "Equity_balance" + " already exists.");
        return (true);
    }
    
    if (StringFind(res, "does not exist") > 0)
    {
        Print("Table " + "Equity_balance" + " does not exist, will create.");
        
        string create_query = "CREATE TABLE \"Equity_balance\" (balance numeric(10,5) NOT NULL, equity numeric(10,5) NOT NULL, \"timestamp\" timestamp with time zone NOT NULL, CONSTRAINT \"Equity_balance_pkey\" PRIMARY KEY (\"timestamp\", balance, equity)) WITH (OIDS=FALSE);";
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

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- create timer
   EventSetTimer(10);
   string res = pmql_connect(g_db_ip_setting, g_db_port_setting, g_db_user_setting, g_db_password_setting, g_db_name_setting);
    if ((res != "ok") && (res != "already connected"))
    {
        Print("DB not connected!");
        return (-1);
    }

    if (!prepare_db())
        return (-1);
    
    return(0);
   
//---
  
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//--- destroy timer
   EventKillTimer();
   pmql_disconnect();
    return(0);
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
   
  }
//+------------------------------------------------------------------+
//| Timer function                                                   |
//+------------------------------------------------------------------+
void OnTimer()
  {
//---

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

    string query = "INSERT INTO \"Equity_balance\" (timestamp, balance, equity) VALUES ('" + time + g_timezone_setting + "', " + DoubleToStr(NormalizeDouble(AccountBalance(), 5), 5) + ", " + DoubleToStr(NormalizeDouble(AccountEquity(), 5), 5) + ")";
    string res = pmql_exec(query);

    if (is_error(res))
        Print(res);
        
    query = "SELECT * FROM \"Equity_balance\" ORDER BY \"timestamp\" DESC LIMIT 50" ;
    res = pmql_exec(query);
   
   if (is_error(res))
   {
      Print("Error executing query: ", res);
      return;
   }
   
   // Split the result into rows
   string rows[];
   StringSplit(res, '*', rows);
   
   // Print the rows
   for (int i = 0; i < ArraySize(rows); i++)
   {
      string row = rows[i];
      if (StringLen(row) > 0)
      {
         
         Print("Row ", i, ": ", row);
      }
   }

    return(0);
   
  }
//+------------------------------------------------------------------+
