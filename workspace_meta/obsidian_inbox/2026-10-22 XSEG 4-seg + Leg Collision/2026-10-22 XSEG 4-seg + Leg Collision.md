# 2026-10-22 XSEG: מפת 4 סגמנטים + Leg Collision

המשך של [[2026-10-19 XSEG cross-segment feature RKD]], [[2026-10-21 Fine segments]]

## מטרה
- XSEG על מפת VLPart גסה (partimagenet ≥0.3) + arm חדש `full` (בלי detach).
- דוגמה שנייה: סוס 164135 (Leg Collision).

## 56069, מפת 4 סגמנטים
- שגיאה: ZS 10.3 | perseg 11.5 | table 10.9 | wonly 16.6 | Fonly 12.0 | full 10.9. אף arm לא מתחת ל-ZS.

![[lvis56069_xseg_vlpart4_backproj_all.jpg]]

## 164135: השלב הגלובלי נחוץ
- two-stage: הרגל כולה על הצד הנכון (2% mixed). per-seg ו-XSEG Fonly: גשר (23–29%).

![[lvis164135_legcollision_zs_twostage_perseg_fonly.jpg]]

## 164135: XSEG על המפה הישנה ועל VLPart
| arm | ישנה | VLPart |
|---|---|---|
| two-stage | 6.4 | 6.4 |
| perseg | 6.3 | 2.9 |
| table | 9.2 | 5.9 |
| wonly | 10.7 | 19.5 |
| Fonly / full | 6.5 | 2.9 |

![[lvis164135_xseg_backproj_compare.jpg]]

- הרגל: כל ה-arms של XSEG נשארים עם גשר (42–66% mixed).

![[lvis164135_xseg_vlpart8_leg_seg3.jpg]]

## מסקנות
- VLPart מוריד את השגיאה לחצי, אבל לא פותר את הרגל.
- XSEG לא לומד יחסים בין סגמנטים: הוא עובד לפי features ולא לפי מסלולים, והרגליים דומות ב-features (טל).
- מבין ה-arms של XSEG, Fonly ו-full הכי סבירים (טל).

## הבא
- טל מסכם ומחליט על כיוון. אפשרות: two-stage על מפת VLPart.
