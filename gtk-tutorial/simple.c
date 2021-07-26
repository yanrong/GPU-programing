#include <gtk/gtk.h>

int main(int argc, char *argv[])
{
	GtkWidget *window; //declare a base widget
	gtk_init(&argc, &argv); // init gtk application run enviroment

	window = gtk_window_new(GTK_WINDOW_TOPLEVEL); //create a new window
	gtk_widget_show(window); //display the window created before
	// Connects a GCallback function to a signal for a particular object
	g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);

	gtk_main();
	return 0;
}