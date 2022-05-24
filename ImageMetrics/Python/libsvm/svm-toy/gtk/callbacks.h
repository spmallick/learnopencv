#include <gtk/gtk.h>

#ifdef __cplusplus
extern "C" {
#endif

void
on_window1_destroy                     (GtkObject       *object,
                                        gpointer         user_data);

gboolean
on_draw_main_button_press_event        (GtkWidget       *widget,
                                        GdkEventButton  *event,
                                        gpointer         user_data);

gboolean
on_draw_main_expose_event              (GtkWidget       *widget,
                                        GdkEventExpose  *event,
                                        gpointer         user_data);

void
on_button_change_clicked               (GtkButton       *button,
                                        gpointer         user_data);

void
on_button_run_clicked                  (GtkButton       *button,
                                        gpointer         user_data);

void
on_button_clear_clicked                (GtkButton       *button,
                                        gpointer         user_data);

void
on_button_save_clicked                 (GtkButton       *button,
                                        gpointer         user_data);

void
on_button_load_clicked                 (GtkButton       *button,
                                        gpointer         user_data);

void
on_fileselection_destroy               (GtkObject       *object,
                                        gpointer         user_data);

void
on_filesel_ok_clicked                  (GtkButton       *button,
                                        gpointer         user_data);

void
on_filesel_cancel_clicked              (GtkButton       *button,
                                        gpointer         user_data);
#ifdef __cplusplus
}
#endif
