# Debug log — dino encoder training produces bad results

## הקשר
הרצנו pipeline של encoder.py על guitar.obj, עם model_type=dino:

```bash
# שלב 1: יצירת random views
python encoder.py --obj_path ./meshes/guitar.obj --name guitar --encoder_data_dir ./data/guitar/encoder_data \
  --generate_random_views 1 --encode_random_views 0 --start_training 0 --test 0

# שלב 2: encode + training
python encoder.py --obj_path ./meshes/guitar.obj --name guitar --encoder_data_dir ./data/guitar/encoder_data \
  --encoder_model_dir ./experiments/guitar/encoder_dino \
  --model_type dino --width 1024 --n_classes 1024 \
  --generate_random_views 0 --encode_random_views 1 --start_training 1

# שלב 3: בדיקה
python image_shape_correspondence.py --mode 'bsb' --img_path ./images/guitar_01.png \
  --object_pixel 130 140 --object_mask_idx 2 --test_pixel 130 140 --test_mask_idx 0 \
  --save_dir ./demo/guitar/bsb/ --obj_path ./meshes/guitar.obj \
  --encoder_dino_f_path ./demo/guitar/encoder_dino/pred_f.pth \
  --encoder_sam_f_path ./demo/guitar/encoder_sam/pred_f.pth --model_name decoder_checkpoint.pth
```

התוצאות של שלב 3 יצאו גרועות. השערת המוצא: אחד השינויים שלנו (מסומנים `--- Tal edit start/end ---` ב-git diff) פגע באימון.

## המאגר
`/workspace/bsb` — git repo, remote `github.com/threedle/bsb`. `git diff` מראה בדיוק 3 שינויים שלנו, בשני קבצים:

1. **`encoder.py`** — בסוף `generate_random_views`: הוספת שמירה של `random_viewing_angles.pt` (elev_list/azim_list). בקוד המקורי זה לא נשמר בכלל, מה שגרם ל-`FileNotFoundError` בכל שלב הבא.
2. **`encoder.py`** — ב-`__main__`: שינוי `train_encoder(args, sam)` ל-`train_encoder(args, model)`. במקור `sam` לא מוגדר כש-`model_type=='dino'` → `NameError`.
3. **`modules/dino.py`** — טעינת dinov2 מ-checkout מקומי מוצמד ל-commit `9c7e324` (ה-commit הראשון עם ה-backbones `_reg`) במקום `torch.hub.load('facebookresearch/dinov2', ...)`, כי הראשי (`main`) משתמש ב-syntax של Python 3.10 ששובר על הסביבה שלנו (3.9).

## מסקנות עד כה — בדיקת כל אחד מ-3 השינויים

| שינוי | מסקנה |
|---|---|
| שמירת `random_viewing_angles.pt` | **תקין, הכרחי.** יש רק עותק אחד בכל הריפו (`data/guitar/encoder_data/224/random_viewing_angles.pt`, נוצר 24/7 09:51). התמונות (`target_save_i.png`) והזוויות נוצרות באותה איטרציית לולאה — אין בעיית סנכרון. אין שום `dino_f`/זוויות "ישנות" אחרות בריפו שהיה צריך להשתמש בהן במקומן. |
| `train_encoder(args, model)` במקום `train_encoder(args, sam)` | **ניטרלי לחלוטין** — הפרמטר השני של `train_encoder` לא בשימוש בכלל בתוך גוף הפונקציה. |
| dinov2 מ-checkout מקומי מוצמד | `embed_dim=1024` בקוד המקומי תואם בדיוק ל-`--width 1024 --n_classes 1024`. הטעינה של המשקולות המאומנות נעשית עם `strict=True` מול ה-URL הרשמי — אם זה לא נכשל, הארכיטקטורה מדויקת. **סביר שלא זה הבעיה.** |

## הממצא המרכזי (לא קשור לעריכות שלנו!) — `modules/dataset.py`

`EncoderDataset` (קובץ **לא נגוע** ע"י Tal edits) הרדקד לגמרי לתיקיית `sam_f` ולמפתח `'sam_f'`, בלי קשר ל-`args.model_type`:

- [modules/dataset.py:50-62](modules/dataset.py#L50-L62) — `prepare_path_list` תמיד עושה `glob(data_dir/'sam_f'/render_res/'*')`.
- [modules/dataset.py:39-47](modules/dataset.py#L39-L47) — `_load_data` תמיד קורא `file['sam_f']`.

בפועל:
- `data/guitar/encoder_data/sam_f/224/` **ריקה לגמרי** (0 קבצים) — מעולם לא הרצנו `encode_random_views` עם `model_type=sam`.
- קובץ ה-cache `data/guitar/encoder_data/sam_f/encoder_samf_path.pt` (נוצר 24/7 10:19 — **בדיוק** מועד השמירה של `pred_f.pth`) מכיל **רשימה ריקה `[]`**.

**המשמעות:** ה-`DataLoader` באימון קיבל 0 batches. הלולאה `for i, (...) in enumerate(tqdm(dataloader))` לא ביצעה אף איטרציה — בלי שגיאה — לאורך כל ה-epochs. `pred_f.pth` שנשמר בסוף מגיע מ-encoder שמעולם לא עבר צעד gradient אחד (עדיין באתחול אקראי).

**זו ההסבר הסביר ביותר לתוצאות הגרועות** — לא בעיה עדינה באימון, אלא שהאימון בפועל לא קרה בכלל.

בונוס: גם אם `sam_f/224` הייתה מלאה — `encode_random_views` תמיד שומר את הפיצ'רים תחת מפתח `'model_f'` ([encoder.py:247](encoder.py#L247)), לא `'sam_f'`, כך שגם מסלול ה-SAM המקורי (הלא ערוך) היה נופל על `KeyError`. נראה שהקוד הזה מעולם לא נבדק סוף-עד-סוף עבור dino (וכנראה גם לא עבור sam).

## ממצא משני — שימו לב לנתיב ב-test command
`demo/guitar/encoder_dino/pred_f.pth` (הנתיב שמשמש ב-`image_shape_correspondence.py` בשלב 3) הוא **קובץ אחר** מ-`experiments/guitar/encoder_dino/pred_f.pth` (הפלט בפועל של האימון):

```
demo/guitar/encoder_dino/pred_f.pth        md5: 8bb7b2b8...   mtime: 14 באפריל (המקורי מה-demo, מעולם לא נדרס)
experiments/guitar/encoder_dino/pred_f.pth md5: ea2cd6e0...   mtime: 24 ביולי (הפלט של האימון שלנו)
experiments/guitar/encoder_dino2/pred_f.pth md5: ea2cd6e0...  mtime: 25 ביולי (זהה bit-for-bit לריצה הקודמת — האימון דטרמיניסטי)
```

כלומר גם לפני שמתקנים את הבאג ב-`dataset.py`, שלב הבדיקה (שלב 3) הצביע על הקובץ הלא נכון — צריך להצביע ל-`experiments/guitar/encoder_dino/pred_f.pth`.

## Next steps (לא בוצע עדיין)
1. לתקן את `modules/dataset.py` כך ש-`EncoderDataset` ישתמש ב-`args.model_type` במקום הרדקוד `'sam_f'` — גם בנתיב התיקייה (`prepare_path_list`) וגם במפתח בתוך ה-pt (`_load_data`, במקום `file['sam_f']`).
2. למחוק/לרענן את קובץ ה-cache הריק `data/guitar/encoder_data/sam_f/encoder_samf_path.pt` (או המקביל אחרי התיקון) כדי שהוא ייבנה מחדש נכון.
3. להריץ מחדש את שלב 2 (encode_random_views + start_training) ולוודא שה-loss בפועל יורד (יהיו הדפסות "Last 20 MSE score" — אם לא מודפס כלום, ה-DataLoader עדיין ריק).
4. בשלב 3, להצביע `--encoder_dino_f_path` ל-`./experiments/guitar/encoder_dino/pred_f.pth` (הקובץ שהאימון בפועל הפיק), לא ל-`./demo/guitar/encoder_dino/pred_f.pth`.

## המשך דיון (25/7, המשך מהשיחה למעלה)

**שאלת Tal:** SAM הוא סגמנטור קפוא (frozen) שרק צריך להשתמש בו מ-checkpoint קיים — לא לאמן אותו. האם התיקון המוצע ב-`dataset.py` עדיין נכון לאור זה?

**תשובה/הבהרה שהתקבלה (טרם אומתה בפועל בדיבאגר):**
יש שני מודלים נפרדים לגמרי בקוד:
1. **SAM עצמו** — קפוא, נטען מ-checkpoint רק אם `args.model_type=='sam'` ([encoder.py:450-451](encoder.py#L450-L451)). בריצה שלנו (`model_type=dino`) הוא **כלל לא נטען** — אין קשר בין הבאג לאימון/שימוש ב-SAM.
2. **"Encoder" הקטן** ([encoder.py:300-301](encoder.py#L300-L301), `train_encoder`) — רשת per-shape שכן מאמנים, ולומדת למפות vertices למרחב הפיצ'רים של המודל הנבחר (dino/sam לפי `args.model_type`). ה-loss ([encoder.py:342](encoder.py#L342)) משתמש במשתנה בשם `batch_sam_f` שמקורו מ-`EncoderDataset`/`dataloader` — זה רק שם ישן (מהגרסה שתמכה רק ב-SAM), לא אינדיקציה שמדובר בפיצ'רי SAM בפועל.

המסקנה: הבאג הוא ב-`EncoderDataset` (הקוד הכללי, לא ערוך על ידינו) שקורא תמיד מתיקיית `sam_f` הריקה במקום `dino_f` המלאה, בלי קשר ל-`args.model_type`. זה לא סותר את ההנחיה לגבי SAM — ה"Encoder" הקטן הזה שונה לגמרי מ-SAM עצמו.

**עדיין לא אומת:** Tal רוצה להריץ עם דיבאגר ולראות בפועל שה-DataLoader/dataset מקבל 0 דגימות, לפני שמבצעים את התיקון. **התיקון ב-`modules/dataset.py` עדיין לא בוצע** — אין שינוי בפועל בקובץ הזה נכון לעכשיו.

### להמשך מחר
- להריץ עם דיבאגר (breakpoint ב-`EncoderDataset.__init__`/`prepare_path_list` או ב-`train_encoder` לפני הלולאה) ולוודא בפועל ש-`len(dataset)==0` / ש-`encoder_samf_path.pt` נטען כרשימה ריקה.
- אם מאומת → להמשיך לפי ה-next steps 1-4 למעלה (עדיין לא בוצעו).

## אימות נוסף (25/7, שיחה נפרדת — עדיין בלי דיבאגר)

חזרנו על כל הממצאים העיקריים מקריאת קבצים ישירה (`torch.load` + `find`/`ls`), בלי שינוי קוד:

- `random_viewing_angles.pt`: `elev`/`azim` הם רשימות של 1000 tensors כל אחד `shape=(1,)`, תואם בדיוק ל-1000 קבצי `target_save_i.png` (0..999). **אין בעיה בקובץ הזה.**
- `data/guitar/encoder_data/sam_f/224/` — עדיין 0 קבצים.
- `data/guitar/encoder_data/sam_f/encoder_samf_path.pt` — עדיין `[]` (רשימה ריקה, `torch.load` מאשר).
- `data/guitar/encoder_data/dino_f/224/target_dino_f_0.pt` — נבדק ישירות: `dict_keys(['model_f', 'elev', 'azim'])`, `model_f.shape == (1, 1024, 64, 64)`. מאשר שאין מפתח `'sam_f'` בכלל בקבצים האלה.

כלומר ההשערה מהניתוח הקודם (עמ' 42-57) מאומתת שוב באופן עצמאי, בלי לגעת עדיין בדיבאגר. התיקון ב-`modules/dataset.py` **עדיין לא בוצע**.
