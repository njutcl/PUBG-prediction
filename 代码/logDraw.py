from chicken_dinner.pubgapi import PUBG

api_key = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJqdGkiOiI1ODYxYWU3MC04MDk5LTAxMzgtNGEzYS0wMDkzNGM5NGQzOGEiLCJpc3MiOiJnYW1lbG9ja2VyIiwiaWF0IjoxNTkwMzk5NDc5LCJwdWIiOiJibHVlaG9sZSIsInRpdGxlIjoicHViZyIsImFwcCI6InNjdWx0Yy1nbWFpbC1jIn0.wyTw_Zm0HXcE2Vs6k1udjWO-dMerDdQQ6eyVxpx7Wkk"
pubg = PUBG(api_key, "pc-na")
shroud = pubg.players_from_names("shroud")[0]
recent_match_id = 'c2c47284-b039-4ccf-b7dc-859cc992ce3d'
recent_match = pubg.match(recent_match_id)
recent_match_telemetry = recent_match.get_telemetry()
recent_match_telemetry.playback_animation("CallmeShenLe1.html",label_players=['CallmeShenLe'],highlight_teams=['CallmeShenLe'],zoom=True,damage=True,use_hi_res=True,dead_player_labels=['CallmeShenLe','linlin2012C'] )