# 2026-10-22 ‏Feature RKD ablation (pod 53)
המשך של `2026-10-19 XSEG cross-segment feature RKD`.

## מטרה
- לבדוק אם ה-feature RKD תורם בכלל. ההבדל בין perseg ל-XSEG הוא לא רק feature: ב-perseg יש 16 זוגות in-seg, וב-XSEG יש 8 in-seg ועוד 8 cross שהולכים ל-feature.

## ניסויים (Fonly, אותן מפות VLPart כמו ב-pod 52)
- **nofeat:** ה-split של XSEG בלי ה-feature term, כלומר 8 זוגות in-seg spatial ועוד cycle.
- **featonly:** ‏Fonly על הזוגות ה-cross ועוד cycle, בלי spatial.

| arm | 56069 | 164135 | רגל 164135 x- / x+ / mixed |
|---|---|---|---|
| ZS | 10.3 | 13.3 | 37 / 63 / 61 |
| two-stage | 39.3 | 6.4 | 100 / 0 / 2 |
| perseg | 11.5 | 2.9 | 24 / 76 / 46 |
| nofeat | 10.6 | 2.9 | 28 / 72 / 55 |
| Fonly | 12.0 | 2.9 | 46 / 54 / 45 |
| featonly | 26.1 | 11.6 | 11 / 89 / 22 |

![[lvis56069_featablation_backproj.png]]
![[lvis164135_featablation_backproj.png]]
![[lvis164135_featablation_leg_seg3.jpg]]

## מסקנות
- בשני הסוסים nofeat ו-Fonly שווים בטווח ה-noise, כלומר ה-feature term לא תורם.
- featonly בלי spatial מתפרק.
- מי שעושה את העבודה הוא ה-in-segment spatial. רק two-stage מפריד את הרגל.
- דוגמה אחת לכל סוס ו-seed אחד.

## הבא
- טל בוחן את התוצאות.
- אם מורידים את ה-feature RKD: לחפש teacher ל-cross-segment, למשל two-stage על מפת VLPart, ‏geodesic כ-default, או כלל גנרי ל-segments.
