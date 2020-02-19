#ifndef SELECTION_WIDGET_H
#define SELECTION_WIDGET_H

#include <QWidget>
#include <QLabel>
#include <QLineEdit>


namespace bsaund_shape_completion
{
    class SelectionWidget: public QWidget
    {

        int current_index = 0;
        int increment = 1;

        Q_OBJECT
    public:
        // This class is not instantiated by pluginlib::ClassLoader, so the
        // constructor has no restrictions.
        SelectionWidget( QWidget* parent = 0 );
        QLabel *display_label, *display_index;
        QLineEdit *increment_editor;

        // We emit outputVelocity() whenever it changes.
    Q_SIGNALS:
        void requestNewFile(std::string filename);
            
    protected:
        void updateLabel();

    protected Q_SLOTS:
        void next();
        void prev();
        void updateIncrement();
    };

} // end namespace bsaund_shape_completion


#endif // DRIVE_WIDGET_H
